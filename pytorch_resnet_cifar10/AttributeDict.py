from collections import OrderedDict

class AttributeDict(OrderedDict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
