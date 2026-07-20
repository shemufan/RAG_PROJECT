import unittest

from pydantic import ValidationError

from backend.schemas.classify_schema import ClassificationOutput, FieldProfile


class SchemaTests(unittest.TestCase):
    def test_field_profile_accepts_minimal_request(self):
        profile = FieldProfile(field_name="user_id")
        self.assertEqual(profile.field_name, "user_id")
        self.assertEqual(profile.sample_values, [])
        self.assertEqual(profile.business_domain, "general")

    def test_field_profile_rejects_blank_field_name(self):
        with self.assertRaises(ValidationError):
            FieldProfile(field_name="   ")

    def test_classification_output_validates_level_and_confidence(self):
        with self.assertRaises(ValidationError):
            ClassificationOutput(
                category="个人信息",
                level="L5",
                confidence=1.2,
                reason="invalid",
                need_review=False,
            )


if __name__ == "__main__":
    unittest.main()
