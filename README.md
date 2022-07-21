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

You should change sqlite path at `code/LoadData_lab.py`  in `def jobid_search_pyqum(id)`

You can import the  `AutoScan1Q_classfile.py` , and call the class `AutoScan1Q`

```python
class AutoScan1Q:
    def __init__(self,numCPW,sparam,dcsweepch):
        self.jobid_dict = {"CavitySearch":0,"PowerDepend":0,"FluxDepend":0,"QubitSearch":0}
        self.sparam = sparam
        self.dcsweepch = dcsweepch
        try:
        	self.numCPW = int(numCPW)
        except:
            pass
    def cavitysearch(self):
        jobid = Quest_command(self.sparam).cavitysearch(self.dcsweepch)
        self.jobid_dict["CavitySearch"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.cavity_list = CavitySearch(dataframe).do_analysis(self.numCPW)
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



### Step 0. Initial your machine parameter

```python
print("start gogogo\n")
routine = AutoScan1Q(numCPW = "3",sparam="S21,",dcsweepch = "1") #sparam initial is "S21,", dcsweepch initial is "1"
```

##### **First of all**

```python
def __init__(self,numCPW,sparam,dcsweepch):
```

We need to define the `numCPW` which means the number of CPW in original chip design, we also can pass it if you don't have.

And `sparam` is meaning the measurement way like "S21," or "S43,"

`dcsweepch` is meaning the signal port of DC-sweep channel.

### Step 1. Cavity Search

```python
print("start step1\n")
routine.cavitysearch()
print(routine.cavity_list)
print(routine.total_cavity_list)
```

Check the bare cavity frequency in 3 to 9 Ghz (3000points)

### Step 2. PowerDepend + FluxDepend + Qubitsearch

```python
for i in routine.total_cavity_list:
    print("start step2\n")
    routine.powerdepend(i)
    f_bare = mean(routine.cavity_list[str(i)])
    print("start step3\n")
    routine.fluxdepend(i,f_bare)
    print("start step4\n")
    routine.qubitsearch(i)
    break #test once
```

Because this section need to use cavity one by one, we call the "for loop" to do the analysis step by step.
