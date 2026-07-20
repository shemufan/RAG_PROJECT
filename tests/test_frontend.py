import httpx
import unittest

from frontend.app import classify_via_api


class FrontendTests(unittest.TestCase):
    def test_formats_success_response(self):
        def handler(request):
            return httpx.Response(
            200,
            json={
                "code": 200,
                "message": "success",
                "data": {
                    "field_name": "id_card",
                    "category": "敏感个人信息",
                    "subcategory": "身份标识",
                    "level": "L4",
                    "confidence": 0.92,
                    "reason": "测试理由",
                    "evidence": [
                        {
                            "content": "第二十八条内容",
                            "source": "个人信息保护法.txt",
                            "article": "第二十八条",
                            "score": 0.9,
                        }
                    ],
                    "need_review": False,
                    "decision_path": "rag_llm",
                },
            },
            )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = classify_via_api("id_card", "身份证号", "用户身份证号码", "", client=client)
        self.assertEqual(result[0], "L4")
        self.assertIn("敏感个人信息", result[1])
        self.assertIn("个人信息保护法.txt", result[4])

    def test_reports_connection_error(self):
        def handler(request):
            raise httpx.ConnectError("offline", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        result = classify_via_api("id_card", "", "", "", client=client)
        self.assertEqual(result[0], "连接失败")
        self.assertIn("后端", result[2])


if __name__ == "__main__":
    unittest.main()
