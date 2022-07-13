# AutoScan1Q
by 朱宗彥, 吳岱家

## Prerequisites
- Python3
- numpy, matplotlib
- pandas
- sklearn, keras
- kneed
- sqlite3
- [resonator_tools](https://github.com/sebastianprobst/resonator_tools) (whole in folder tools)

## How to use
You can import the  `Classfile.py` , and call the class `AutoScan1Q`

```
class AutoScan1Q
    def __init__(self,numCPW)
    def cavitysearch(self)
    def powerdepend(self)
    def fluxdepend(self, f_bare)
    def qubitsearch(self)
```

Step by step to do the whole measurement (cavitysearch > powerdepend > fluxdepend > qubitsearch)