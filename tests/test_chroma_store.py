import unittest

from backend.storage.chroma_store import lexical_score


class ChromaStoreTests(unittest.TestCase):
    def test_lexical_score_prefers_matching_rule(self):
        query = "field_name: product_name\nfield_cn: 商品名称\nbusiness_domain: product"
        matching = "规则 10：一般业务数据，适用字段：商品名称、公开产品信息"
        unrelated = "个人信息处理者应当保护身份证件号码"

        self.assertGreater(lexical_score(query, matching), lexical_score(query, unrelated))
        self.assertGreater(lexical_score(query, matching), 0)


if __name__ == "__main__":
    unittest.main()
