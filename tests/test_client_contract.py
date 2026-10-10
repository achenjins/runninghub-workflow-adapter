"""RunningHubClient contract tests against response bodies taken from the platform and its docs."""
import unittest
from unittest.mock import Mock, patch

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
            with patch("rh_generic_lib.runninghub_client.requests.post", return_value=response(body)) as post:
                with self.assertRaises(RunningHubError) as caught:
                    await client().wait_for_result("1904152026220003329")
            self.assertNotIsInstance(caught.exception, RunningHubTransportError)
            self.assertIn(body["errorCode"], str(caught.exception))
            self.assertEqual(post.call_count, 1)

    async def test_missing_task_is_terminal(self):
        for body in (QUERY_807, QUERY_1004):
            with patch("rh_generic_lib.runninghub_client.requests.post", return_value=response(body)):
                with self.assertRaises(RunningHubTaskError) as caught:
                    await client().wait_for_result("1")
            self.assertEqual(caught.exception.status, "FAILED")

    async def test_rate_limit_is_retried(self):
        replies = [response(QUERY_1003), response(QUERY_SUCCESS)]
        with patch("rh_generic_lib.runninghub_client.requests.post", side_effect=replies), \
                patch("rh_generic_lib.runninghub_client.asyncio.sleep"):
            result = await client().wait_for_result("1")
        self.assertEqual(result["status"], "SUCCESS")

    async def test_running_then_success(self):
        replies = [response(QUERY_RUNNING), response(QUERY_SUCCESS)]
        with patch("rh_generic_lib.runninghub_client.requests.post", side_effect=replies), \
                patch("rh_generic_lib.runninghub_client.asyncio.sleep"):
            result = await client().wait_for_result("1")
        self.assertEqual(result["results"][0]["url"], "https://rh-images.example/out.jpg")


class SubmitContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_codes_are_retryable(self):
        for code, message in (("421", "TASK_QUEUE_MAXED"), ("415", "TASK_INSTANCE_MAXED"), ("1520", "Concurrency limit reached")):
            body = {"taskId": "", "status": "", "errorCode": code, "errorMessage": message, "results": None}
            with patch("rh_generic_lib.runninghub_client.requests.post", return_value=response(body)):
                with self.assertRaises(RunningHubBusyError):
                    await client().submit([])

    async def test_payload_follows_documented_parameters(self):
        accepted = {"taskId": "2009191190196789249", "status": "QUEUED", "errorCode": "", "errorMessage": ""}
        expected = {"Standard": None, "Plus": "plus", "Ultra": "ultra"}
        for instance_type, documented in expected.items():
            with patch("rh_generic_lib.runninghub_client.requests.post", return_value=response(accepted)) as post:
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
            with patch("rh_generic_lib.runninghub_client.requests.post", return_value=response(body)):
                with self.assertRaises(RunningHubError) as caught:
                    await client().submit([])
            self.assertNotIsInstance(caught.exception, (RunningHubBusyError, RunningHubTransportError))


if __name__ == "__main__":
    unittest.main()
