import json

class KnowledgeBase:
    def __init__(self, path="data/knowledge_base.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    def get_info(self, fault_class):
        fault_class = str(fault_class)
        if fault_class in self.kb:
            return self.kb[fault_class]
        else:
            return {"name": "Неизвестная ошибка", "recommendation": "Вызвать инженера."}