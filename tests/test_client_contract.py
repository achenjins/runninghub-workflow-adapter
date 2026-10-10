"""RunningHubClient contract tests against response bodies taken from the platform and its docs."""
import unittest
from unittest.mock import MagicMock, Mock, patch

import requests

from rh_generic_lib.runninghub_client import (
    RunningHubBusyError, RunningHubClient, RunningHubError, RunningHubTaskError, RunningHubTransportError)

# POST https://www.runninghub.cn/openapi/v2/query with an invalid key, HTTP 200 (2026-10-09).
QUERY_806 = {"taskId": "1904152026220003329", "status": "", "errorCode": "806",
             "errorMessage": "APIKEY_USER_NOT_FOUND", "results": None, "clientId": "", "promptTips": "",
             "failedReason": {}, "usage": None, "parentTaskId": None, "taskUsageList": None}
QUERY_1007 = {**QUERY_806, "errorCode": "1007", "errorMessage": "Invalid parameters, please check your input"}
QUERY_1003 = {**QUERY_806, "errorCode": "1003", "errorMessage": "Rate limit exceeded, please slow down requests"}
QUERY_807 = {**QUERY_806, "errorCode": "807", "errorMessage": "APIKEY_TASK_NOT_FOUND"}
QUERY_1004 = {**QUERY_806, "errorCode": "1004", "errorMessage": "Task not found, please check the task ID"}
QUERY_RUNNING = {"taskId": "1", "status": "RUNNING", "errorCode": "", "errorMessage": "", "results": None}
QUERY_SUCCESS = {"taskId": "1", "status": "SUCCESS", "errorCode": "", "errorMessage": "",
                 "results": [{"url": "https://rh-images.example/out.jpg", "outputType": "jpg"}]}


def response(body, status=200):
    resp = Mock(status_code=status)
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def client(**kwargs):
    return RunningHubClient(base_url="https://www.runninghub.cn", api_key="key", workflow_id="1",
                            poll_interval=1, max_wait=2, **kwargs)


class QueryContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_code_body_is_a_definite_refusal(self):
        for body in (QUERY_806, QUERY_1007):
            with patch.object(requests.Session, "post", return_value=response(body)) as post:
                with self.assertRaises(RunningHubError) as caught:
                    await client().wait_for_result("1904152026220003329")
            self.assertNotIsInstance(caught.exception, RunningHubTransportError)
            self.assertIn(body["errorCode"], str(caught.exception))
            self.assertEqual(post.call_count, 1)

    async def test_missing_task_is_terminal(self):
        for body in (QUERY_807, QUERY_1004):
            with patch.object(requests.Session, "post", return_value=response(body)):
                with self.assertRaises(RunningHubTaskError) as caught:
                    await client().wait_for_result("1")
            self.assertEqual(caught.exception.status, "FAILED")

    async def test_rate_limit_is_retried(self):
        replies = [response(QUERY_1003), response(QUERY_SUCCESS)]
        with patch.object(requests.Session, "post", side_effect=replies), \
                patch("rh_generic_lib.runninghub_client.asyncio.sleep"):
            result = await client().wait_for_result("1")
        self.assertEqual(result["status"], "SUCCESS")

    async def test_running_then_success(self):
        replies = [response(QUERY_RUNNING), response(QUERY_SUCCESS)]
        with patch.object(requests.Session, "post", side_effect=replies), \
                patch("rh_generic_lib.runninghub_client.asyncio.sleep"):
            result = await client().wait_for_result("1")
        self.assertEqual(result["results"][0]["url"], "https://rh-images.example/out.jpg")

    async def test_failed_and_cancelled_status_are_terminal(self):
        failed = {"taskId": "1", "status": "FAILED", "errorCode": "1000",
                  "errorMessage": "Unknown error, please retry or contact support", "results": None}
        cancelled = {"taskId": "1", "status": "CANCEL", "errorCode": "", "errorMessage": "", "results": None}
        for body, status in ((failed, "FAILED"), (cancelled, "CANCEL")):
            with patch.object(requests.Session, "post", return_value=response(body)):
                with self.assertRaises(RunningHubTaskError) as caught:
                    await client().wait_for_result("1")
            self.assertEqual(caught.exception.status, status)

    async def test_persistent_outage_gives_up_after_retries(self):
        with patch.object(requests.Session, "post", side_effect=requests.ConnectionError("offline")) as post, \
                patch("rh_generic_lib.runninghub_client.asyncio.sleep"):
            with self.assertRaises(RunningHubTransportError):
                await client(query_retries=2).wait_for_result("1")
        self.assertEqual(post.call_count, 3)


class SubmitContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_codes_are_retryable(self):
        for code, message in (("421", "TASK_QUEUE_MAXED"), ("415", "TASK_INSTANCE_MAXED"), ("1520", "Concurrency limit reached")):
            body = {"taskId": "", "status": "", "errorCode": code, "errorMessage": message, "results": None}
            with patch.object(requests.Session, "post", return_value=response(body)):
                with self.assertRaises(RunningHubBusyError):
                    await client().submit([])

    async def test_payload_follows_documented_parameters(self):
        accepted = {"taskId": "2009191190196789249", "status": "QUEUED", "errorCode": "", "errorMessage": ""}
        expected = {"Standard": None, "Plus": "plus", "Ultra": "ultra"}
        for instance_type, documented in expected.items():
            with patch.object(requests.Session, "post", return_value=response(accepted)) as post:
                task_id = await client().submit([{"nodeId": "6", "fieldName": "text", "fieldValue": "cat"}], instance_type=instance_type)
            self.assertEqual(task_id, "2009191190196789249")
            payload = post.call_args.kwargs["json"]
            self.assertIs(payload["usePersonalQueue"], False)
            self.assertIs(payload["addMetadata"], False)
            self.assertEqual(payload.get("instanceType"), documented)
            self.assertTrue(post.call_args.args[0].endswith("/openapi/v2/run/workflow/1"))

    async def test_rejections_are_not_retryable(self):
        for code, message in (("806", "APIKEY_USER_NOT_FOUND"), ("380", "WORKFLOW_NOT_EXISTS")):
            body = {"taskId": "", "status": "", "errorCode": code, "errorMessage": message, "results": None}
            with patch.object(requests.Session, "post", return_value=response(body)):
                with self.assertRaises(RunningHubError) as caught:
                    await client().submit([])
            self.assertNotIsInstance(caught.exception, (RunningHubBusyError, RunningHubTransportError))

    async def test_unconfirmed_submission_is_unknown(self):
        with patch.object(requests.Session, "post", return_value=response({"code": 200})):
            with self.assertRaises(RunningHubTransportError):
                await client().submit([])
        for missing in ({"workflow_id": ""}, {"api_key": ""}):
            options = {"base_url": "https://www.runninghub.cn", "api_key": "key", "workflow_id": "1", **missing}
            with self.assertRaises(RunningHubError):
                await RunningHubClient(**options).submit([])


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_reuse_one_session(self):
        rh = client()
        with patch.object(requests.Session, "post", return_value=response(QUERY_RUNNING)) as session_post, \
                patch("requests.post") as module_post:
            await rh.query("1")
            await rh.query("1")
        self.assertEqual(session_post.call_count, 2)
        module_post.assert_not_called()

    async def test_failed_download_closes_streamed_response(self):
        streamed = MagicMock(status_code=404)
        streamed.__enter__.return_value = streamed
        streamed.raise_for_status.side_effect = requests.HTTPError("404")
        with patch.object(requests.Session, "get", return_value=streamed), patch("requests.get", return_value=streamed):
            with self.assertRaises(RunningHubError):
                await client().download_bytes("https://rh-images.example/missing.png")
        streamed.__exit__.assert_called_once()

    async def test_http_errors_are_classified(self):
        def http_error(status):
            resp = response({}, status)
            resp.raise_for_status.side_effect = requests.HTTPError(str(status), response=resp)
            return resp
        with patch.object(requests.Session, "post", return_value=http_error(401)):
            with self.assertRaises(RunningHubError) as caught:
                await client().query("1")
        self.assertNotIsInstance(caught.exception, RunningHubTransportError)
        for failure in (http_error(502), requests.Timeout("slow")):
            kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
            with patch.object(requests.Session, "post", **kwargs):
                with self.assertRaises(RunningHubTransportError):
                    await client().query("1")

    async def test_download_respects_size_limit(self):
        streamed = MagicMock(status_code=200)
        streamed.__enter__.return_value = streamed
        streamed.iter_content.return_value = [b"\x89PNG", b"", b"data"]
        with patch.object(requests.Session, "get", return_value=streamed):
            self.assertEqual(await client().download_base64("https://rh-images.example/out.png"), "iVBOR2RhdGE=")
            with self.assertRaises(RunningHubError):
                await client(max_file_bytes=4).download_bytes("https://rh-images.example/out.png")


class AuxiliaryEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_returns_platform_body(self):
        for body in ({"code": 0, "msg": "success", "data": None},
                     {"code": 806, "msg": "APIKEY_USER_NOT_FOUND", "data": None},
                     {"code": 807, "msg": "APIKEY_TASK_NOT_FOUND", "data": None}):
            with patch.object(requests.Session, "post", return_value=response(body)) as post:
                self.assertEqual(await client().cancel("1904152026220003329"), body)
            self.assertTrue(post.call_args.args[0].endswith("/task/openapi/cancel"))
            self.assertEqual(post.call_args.kwargs["json"], {"apiKey": "key", "taskId": "1904152026220003329"})

    async def test_upload(self):
        documented = {"code": 0, "message": "success", "data": {
            "type": "image", "download_url": "https://rh-images.example/input/openapi/abc.png",
            "fileName": "openapi/abc.png", "size": "3490"}}
        with patch.object(requests.Session, "post", return_value=response(documented)) as post:
            self.assertEqual(await client().upload_file(b"\x89PNG", "input.png"), "openapi/abc.png")
        self.assertTrue(post.call_args.args[0].endswith("/openapi/v2/media/upload/binary"))
        self.assertEqual(post.call_args.kwargs["files"], {"file": ("input.png", b"\x89PNG")})
        for body in ({"code": 1, "message": "upload failed", "data": None}, {"code": 0, "message": "success", "data": {}}):
            with patch.object(requests.Session, "post", return_value=response(body)):
                with self.assertRaises(RunningHubError):
                    await client().upload_file(b"\x89PNG", "input.png")
        with self.assertRaises(RunningHubError):
            await client(max_file_bytes=2).upload_file(b"\x89PNG", "input.png")

    async def test_get_workflow_json(self):
        body = {"code": 0, "msg": "success", "data": {"prompt": '{"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "cat"}}}'}}
        with patch.object(requests.Session, "post", return_value=response(body)) as post:
            workflow = await client().get_workflow_json("1904136902449209346")
        self.assertEqual(workflow["6"]["inputs"]["text"], "cat")
        self.assertEqual(post.call_args.kwargs["json"], {"apiKey": "key", "workflowId": "1904136902449209346"})
        self.assertEqual(post.call_args.kwargs["headers"]["Host"], "www.runninghub.cn")
        for bad in ({"code": 380, "msg": "WORKFLOW_NOT_EXISTS", "data": None}, {"code": 0, "msg": "success", "data": {}}):
            with patch.object(requests.Session, "post", return_value=response(bad)):
                with self.assertRaises(RunningHubError):
                    await client().get_workflow_json("1")


if __name__ == "__main__":
    unittest.main()
