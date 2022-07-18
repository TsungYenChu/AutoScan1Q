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


You can import the  `AutoScan1Q_classfile.py` , and call the class `AutoScan1Q`

```
class AutoScan1Q:
    def __init__(self,numCPW,sparam):
        self.numCPW = int(numCPW)
        self.jobid_dict = {"CavitySearch":0,"PowerDepend":0,"FluxDepend":0,"QubitSearch":0}
        self.sparam = sparam
    def cavitysearch(self):
        jobid = Quest_command(self.sparam).cavitysearch()
        self.jobid_dict["CavitySearch"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.cavity_list = CavitySearch(dataframe).do_analysis(numCPW)
    def powerdepend(self,cavity_num):
        jobid = Quest_command(self.sparam).powerdepend(...)
        self.jobid_dict["PowerDepend"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.select_power = PowerDepend(dataframe).do_analysis()
    def fluxdepend(self,cavity_num, f_bare):
        jobid = Quest_command(self.sparam).fluxdepend(...)
        self.jobid_dict["FluxDepend"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.wave = FluxDepend(dataframe).do_analysis(f_bare)
    def qubitsearch(self,cavity_num):
        jobid = Quest_command(self.sparam).qubitsearch(...)
        self.jobid_dict["QubitSearch"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.qubit = Db_Scan(dataframe).do_analysis()
```

Step by step to do the whole measurement (cavitysearch > powerdepend > fluxdepend > qubitsearch)



**First of all**

```
def __init__(self,numCPW,sparam):
```

We need to define the `numCPW` which means the number of CPW in original chip design, we also can pass it if you don't have.

And `sparam` is meaning the measurement way like "S21," or "S43,"

