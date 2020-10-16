from enum import Enum


class EmotionTag(Enum):
    joy = "Joy"
    hate = "Hate"
    love = "Love"
    sorrow = "Sorrow"
    anxiety = "Anxiety"
    surprise = "Surprise"
    anger = "Anger"
    anticipation = "Expect"

    @classmethod
    def reverse_member_map(cls):
        return {enum.value: enum for (enum_name, enum) in cls.__members__.items()}