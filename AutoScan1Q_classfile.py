import sys
sys.path.append('./code')

from colorama import Fore, Back
from flask import session
from pyqum import get_db, close_db
from json import dumps
#---------------load package of load_data---------------
from LoadData_lab import jobid_search_pyqum, pyqum_load_data
#---------------load package of cavity search---------------
from CavitySearch import make_amp,make_pha,input_process,output_process,true_alt_info,find_best_ans,db_datamaker,Find_eps,dbscan,predict_dataset,compa_gru_db
from numpy import array,vstack, hstack
from pandas import Series, DataFrame, concat
from keras.models import load_model
from QubitFrequency import colect_cluster,cal_nopecenter,cal_distance,denoise,check_overpower,find_farest,cal_Ec_GHz,freq2idx
#---------------load package of power dependent---------------
from sklearn.cluster import KMeans
from numpy import median
from PowerDepend import outlier_detect, cloc
#---------------load package of flux dependent---------------
from FluxDepend import flux_load_data, fit_sin
#---------------save jobid list in pickle---------------
from pickle import dump,load
#---------------process---------------
from numpy import mean
# from pyqum.directive.characterize import F_Response, CW_Sweep
from random import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Load_From_pyqum:
    def __init__(self, jobid):
        self.jobid = jobid
        self.pyqum_path, self.task = jobid_search_pyqum(self.jobid)
        # self.pyqum_path = 'data/F_Response.pyqum(2)'
        
    def load(self):
        self.amp_data,self.jobid_check  = pyqum_load_data(self.pyqum_path)
        if self.jobid == self.jobid_check:
            print("JOBid ",self.jobid," checked")
        return self.amp_data
            
class CavitySearch:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # ans of prediction
        self.answer = {}
        self.amplitude, self.phase, self.freq = [], [], []
        self.fig = DataFrame()
        self.ans_array = {}

    def do_analysis(self,designed):
        I = self.dataframe['I']
        Q = self.dataframe['Q']
        self.freq = self.dataframe['Frequency']

        self.amplitude = make_amp(I,Q)
        self.phase = make_pha(I,Q)
        self.fig = DataFrame(concat([Series(self.freq),Series(self.amplitude),Series(self.phase)],axis=1))

        # GRU part
        AMP = load_model('./model/GRU_AMP_Accuracy_ 96.63_.h5')
        PHA = load_model('./model/GRU_PHA_Accuracy_ 95.01_.h5')

        amp, pha, comparison = input_process(self.fig)      # frequency,amplitude,phase; comparison[no.][0] for freq start, end for comparison[no.][1] 
        self.fig.columns = ['<b>frequency(GHz)</b>','Amplitude','UPhase']

        # prediction GRU
        amp_pred = AMP.predict(amp)
        pha_pred = PHA.predict(pha)

        # result process
        true_out ,alt = output_process(amp_pred,pha_pred,comparison)  
        zone, voted_amp, voted_pha = true_alt_info(true_out,alt,self.fig)

        gru_ans_amp, status_amp = find_best_ans(zone,voted_amp,self.fig,designed)  # status is the origin predict result with the default peak_limit = 8
        gru_ans_pha, status_pha = find_best_ans(zone,voted_pha,self.fig,designed)
        
        # DBSCAN part
        # dbscan for phase
        inp_db = db_datamaker(self.phase,self.freq)
        eps,mini = Find_eps(inp_db) 
        l_d_pha = dbscan(inp_db,eps,mini)
        ture_out_db_pha = predict_dataset(l_d_pha,self.freq)

        # dbscan for amplitude
        inp_db = db_datamaker(self.amplitude,self.freq)
        eps,mini = Find_eps(inp_db) 
        l_d_amp = dbscan(inp_db,eps,mini)
        ture_out_db_amp = predict_dataset(l_d_amp,self.freq)

        true_out_db = vstack((ture_out_db_amp,ture_out_db_pha))

        zone, voted_amp, voted_pha = true_alt_info(true_out_db,alt,self.fig)
        db_ans_amp, status_amp = find_best_ans(zone, voted_amp, self.fig, designed)
        db_ans_pha, status_pha = find_best_ans(zone, voted_pha, self.fig, designed)

        amp_ans = [gru_ans_amp,db_ans_amp]
        pha_ans = [gru_ans_pha,db_ans_pha]

        self.answer = compa_gru_db(amp_ans,pha_ans)   # answer looks: {'0':[start,end],'1':[...],...}
        return self.answer

    def give_answer_array(self,no):
        self.ans_array = {
        'Frequency':self.fig[self.fig["<b>frequency(GHz)</b>"].between(self.answer[str(no)][0],self.answer[str(no)][1])]['<b>frequency(GHz)</b>'],
        'Amplitude':self.fig[self.fig["<b>frequency(GHz)</b>"].between(self.answer[str(no)][0],self.answer[str(no)][1])]['Amplitude'],
        'UPhase':self.fig[self.fig["<b>frequency(GHz)</b>"].between(self.answer[str(no)][0],self.answer[str(no)][1])]['UPhase']
        }

class PowerDepend:
    def __init__(self, dataframe):
        self.data = loadmat_valid(dataframe)
    def do_analysis(self):
        model = KMeans(n_clusters=2, n_init=1, random_state=0)
        label = model.fit_predict(self.data)
        label_new = outlier_detect(self.data,label)
        power_0,power_1 = cloc(label_new)
        print("power : "+"{:.2f}".format(self.data[:, 0][power_0])+"{:<7}".format(' dBm ; ')+
              "fr : "+"{:.2f}".format(median(self.data[:, 1][label_new ==0]))+"{:<7}".format(' MHz ; \n')+
              "power : "+"{:.2f}".format(self.data[:, 0][power_1])+"{:<7}".format(' dBm ; ')+
              "fr : "+"{:.2f}".format(median(self.data[:, 1][label_new ==1]))+"{:<7}".format(' MHz ; '))
        self.select_power = min(self.data[:, 0][power_0],self.data[:, 0][power_1])
        return self.select_power
        
class FluxDepend:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    def do_analysis(self,f_bare):
        tol = 0.1
        self.valid = flux_load_data(self.dataframe)
        self.valid = self.valid.drop(self.valid[(self.valid['fr']<f_bare+tol) & (self.valid['fr']>f_bare-tol)].index)
        ki = self.valid['fr']-f_bare
        f_qubit = f_bare-1/ki
        offset = self.valid['flux'][f_qubit ==f_qubit.max()]
        f_dress = self.valid['fr'][offset.index]
        res = fit_sin(self.valid['flux'],f_qubit)
        period = float(res['period'])
        print("{:<36}".format("Final_dressed cavity frquency"), " : " , "{:>8.2f}".format(float(f_dress)) ,"MHz")
        print("{:<36}".format("Final_bare cavity frquency"), " : " , "{:>8.2f}".format(float(f_bare)) ,"MHz")
        print("{:<36}".format("Final_dressed cavity frquency diff."), " : " , "{:>8.2f}".format(float(f_dress-f_bare)) ,"MHz")
        print("{:<36}".format("Final_offset")," : ","{:>8.2f}".format(float(offset)),"uA")
        print("{:<36}".format("Final_period")," : ","{:>8.2f}".format(float(period)),"uA")
    #     if plot:
    #         import matplotlib.pyplot as plt
    #         from numpy import linspace
    #         plt.rcParams["figure.figsize"] = [20,10]
    #         plt.subplot(211)
    #         plt.scatter(self.valid['flux'],self.valid['fr'],color='black', marker='o',label='real data')
    #         plt.subplot(212)
    #         plt.scatter(self.valid['flux'],f_qubit,color='r', marker='*',label='f_qubit')
    #         x = linspace(self.valid['flux'].min(),self.valid['flux'].max(),200)
    #         plt.plot(x, res["fitfunc"](x), "r-", label="fit curve", linewidth=2)
    #         plt.xlabel("Flux : uA")
    #         plt.ylabel("Freq : MHz")
    #         # plt.ylim(self.valid['fr'].min()-.20,self.valid['fr'].max()+.20)
    #         plt.legend()
    #         plt.show()
        return {"f_dress":float(f_dress/1000),"f_bare":float(f_bare/1000),"f_diff":float((f_dress-f_bare)/1000),"offset":float(offset),"period":float(period)}
    
class QubitFreq_Scan:
    def __init__(self,dataframe):#,Ec,status,area_Maxratio,density
        self.dataframe = dataframe

        self.fq = 0.0
        self.Ec = 0.0
        self.freq = 0.0
        self.status = 0
        self.target_freq = []
        self.sub = []
        self.title = ''
        self.answer = {} # <- 0630 update
        self.plot_items = {}

    def do_analysis(self):
        self.freq = self.dataframe['Frequency']  #for qubit  <b>XY-Frequency(GHz)</b>
        I = self.dataframe['I']
        Q = self.dataframe['Q']

        inp_db = []
        for i in range(I.shape[0]):
            inp_db.append(list(hstack(([I[i]],[Q[i]]))))

        # start DBSCAN
        eps,min_samples = Find_eps(inp_db)
        labels_db = dbscan(array(inp_db),eps,min_samples)

        # output process
        peak_susp_idx, nope_idx = colect_cluster(labels_db,mode='db')
        nope_center = cal_nopecenter(nope_idx,I,Q)

        # redefine the background
        redef_sub = []
        for i in range(self.freq.shape[0]):
            redef_sub.append(cal_distance([I[i],Q[i]],nope_center))

        self.sub = array(redef_sub)
        self.title = 'Amplitude_Redefined'


        if len(peak_susp_idx) != 0:

            tip = denoise(peak_susp_idx,self.freq,self.sub)
            #filter out the overpower case within +-0.5 std
            overpower,_,_ = check_overpower(tip,self.sub,0.5)

            if overpower == 'safe':
                #farest 3 point in IQ
                denoised_freq = find_farest(tip,nope_center,self.sub,I,Q,self.freq)

                #calculate Ec based on farest
                self.fq, self.Ec, self.status, self.target_freq = cal_Ec_GHz(denoised_freq,self.sub,self.freq)
            else:
                self.fq, self.Ec, self.status, self.target_freq = 0, 0, 0, []
        else:
            self.fq, self.Ec, self.status, self.target_freq = 0, 0, 0, []

        self.answer = {'Fq':self.fq,'Ec':self.Ec,'Status':self.status,'Freqs':self.target_freq} 
        '''status = 0 for 0 peak detected -> overpower with high probability
           status = 1 for 1 peak detected -> so far, a stronger xy-power again
           status = 2 for 2 peak detected'''
        return self.answer
                                                                                         
    def give_result(self):
        farest = freq2idx(self.target_freq,self.freq)[:3]
        self.plot_items = {
            'Targets':self.sub[farest],
            'Targets_Freq':self.freq[farest],
            'Sub_Frequency':self.freq,
            'Substrate':self.sub
        }
      
def char_fresp_new(sparam,freq,powa,flux,dcsweepch = "1",comment = "By bot"):
    # Check user's current queue status:
    if session['run_clearance']:
        print(comment)
        wday = int(-1)
        sparam = sparam   #S-Parameter
        ifb = "50"     #IF-Bandwidth (Hz)
        freq = freq #Frequency (GHz)
        powa = powa    #Power (dBm)
        fluxbias = flux   #Flux-Bias (V/A)
        comment = comment.replace("\"","") #comment
        PERIMETER = {"dcsweepch":dcsweepch, "z-idle":'{}', "sweep-config":'{"sweeprate":0.0001,"pulsewidth":1001e-3,"current":1}'} # DC=YOKO
        CORDER = {'Flux-Bias':fluxbias, 'S-Parameter':sparam, 'IF-Bandwidth':ifb, 'Power':powa, 'Frequency':freq}
        print(CORDER)
        # Start Running:
        # TOKEN = 'TOKEN(%s)%s' %(session['user_name'],random())
        workspace = F_Response(session['people'], corder=CORDER, comment=comment, tag='', dayindex=wday, perimeter=PERIMETER)
        return workspace.jobid_analysis
    else: return show()
def char_cwsweep_new(sparam,freq,powa,flux,dcsweepch = "1",comment = "By bot"):
    # Check user's current queue status:
    if session['run_clearance']:
        print(comment)
        wday = int(-1)
        sparam = sparam   #S-Parameter
        ifb = "50"     #IF-Bandwidth (Hz)
        freq = freq #Frequency (GHz)
        powa = powa    #Power (dBm)
        fluxbias = flux   #Flux-Bias (V/A)
        xyfreq = "OPT,"
        xypowa = "OPT,"
        comment = comment.replace("\"","")
        PERIMETER = {"dcsweepch":dcsweepch, "z-idle":'{}', 'sg-locked': '{}', "sweep-config":'{"sweeprate":0.0001,"pulsewidth":1001e-3,"current":0}'} # DC=YOKO
        CORDER = {'Flux-Bias':fluxbias, 'XY-Frequency':xyfreq, 'XY-Power':xypowa, 'S-Parameter':sparam, 'IF-Bandwidth':ifb, 'Frequency':freq, 'Power':powa}
        print(CORDER)
        # Start Running:
        # TOKEN = 'TOKEN(%s)%s'%(session['user_name'],random())
        workspace = CW_Sweep(session['people'], corder=CORDER, comment=comment, tag='', dayindex=wday, perimeter=PERIMETER)

        return workspace.jobid_analysis
    else: return show()
    
class Quest_command:
    def __init__(self,sparam="S21,"):
        self.sparam = sparam

    def jobnote(JOBID, note):
        '''Add NOTE to a JOB after analyzing the data'''
        if g.user['measurement']:
            try:
                db = get_db()
                db.execute('UPDATE job SET note = ? WHERE id = ?', (note,JOBID))
                db.commit()
                close_db()
                print(Fore.GREEN + "User %s has successfully updated JOB#%s with NOTE: %s" %(g.user['username'],JOBID,note))
            except:
                print(Fore.RED + Back.WHITE + "INVALID JOBID")
                raise
        else: pass
    
    def cavitysearch(self,dcsweepch,add_comment=""):
        jobid = char_fresp_new(sparam=self.sparam,freq = "3 to 9 *3000",powa = "0",flux = "OPT,",dcsweepch = "1",comment = "By bot - step1 cavitysearch "+add_comment)
        return jobid
    def powerdepend(self,select_freq,add_comment=""):
        freq_command = "{} to {} *200".format(select_freq[0],select_freq[1])
        jobid = char_fresp_new(sparam=self.sparam,freq=freq_command,powa = "-50 to 10 * 13",flux = "OPT,",dcsweepch = "1",comment = "By bot - step2 power dependent"+add_comment)
        return jobid
    def fluxdepend(self,select_freq,select_powa,add_comment=""):
        freq_command = "{} to {} *200".format(select_freq[0],select_freq[1])
        jobid = char_fresp_new(sparam=self.sparam,freq=freq_command,powa = select_powa,flux = "-500e-6 to 500e-6 * 50",dcsweepch = "1",comment = "By bot - step3 flux dependent "+add_comment)
        return jobid
    def qubitsearch(self,select_freq,select_flux,dcsweepch,add_comment=""):
        freq_command = "{} to {} *200".format(select_freq[0],select_freq[1])
        jobid = char_cwsweep_new(sparam=self.sparam,freq = freq_command,flux = select_flux,powa = "-10 to 10 *4 ",dcsweepch = dcsweepch,comment = "By bot - step4 qubit search "+add_comment)
        return jobid



class AutoScan1Q:
    def __init__(self,numCPW="3",sparam="S21,",dcsweepch = "1"):
        self.jobid_dict = {"CavitySearch":0,"PowerDepend":0,"FluxDepend":0,"QubitSearch":0}
        self.sparam = sparam
        self.dcsweepch = dcsweepch
        try:
            self.numCPW = int(numCPW)
        except:
            pass
        
    def cavitysearch(self):
        # jobid = Quest_command(self.sparam).cavitysearch(self.dcsweepch)
        jobid = 5094
        print("do measurement\n")
        self.jobid_dict["CavitySearch"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        # self.cavity_list = CavitySearch(dataframe).do_analysis(self.numCPW) #model h5 cannot import
        self.cavity_list = {'7116.0 MHz': [7.102, 7.128], '6334.0 MHz': [6.32, 6.346]}
        self.total_cavity_list = list(self.cavity_list.keys())
    def powerdepend(self,cavity_num):
        # jobid = Quest_command(self.sparam).powerdepend(select_freq=self.cavity_list[cavity_num],add_comment="with Cavity "+str(cavity_num))
        jobid = 5097
        self.jobid_dict["PowerDepend"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.select_power = PowerDepend(dataframe).do_analysis() #pass
        print("Select Power : %f"%self.select_power)
    def fluxdepend(self,cavity_num, f_bare):
        # jobid = Quest_command(self.sparam).fluxdepend(select_freq=self.cavity_list[cavity_num],select_powa=self.select_power,add_comment="with Cavity "+str(cavity_num))
        jobid = 5105
        self.jobid_dict["FluxDepend"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.wave = FluxDepend(dataframe).do_analysis(f_bare) #pass
        print(self.wave)
    def qubitsearch(self,cavity_num):
        # jobid = Quest_command(self.sparam).qubitsearch(select_freq=self.cavity_list[cavity_num],select_flux=str(self.wave["offset"])+'e-6',dcsweepch = self.dcsweepch,add_comment="with Cavity "+str(cavity_num))
        jobid = 5106
        self.jobid_dict["QubitSearch"] = jobid
        dataframe = Load_From_pyqum(jobid).load()
        self.qubit = QubitFreq_Scan(dataframe).do_analysis() #examine the input data form is dataframe because Series cannot reshape 
        print(self.qubit)

def save_class(item,path = "save.pickle"):
    with open(path, 'wb') as f:
        dump(item, f)
def load_class(path = "save.pickle"):
    with open(path, 'rb') as f:
        item = load(f)
    return item


if __name__ == "__main__":
    print("start gogogo\n")
    # search(self.quantificationObj)
    routine = AutoScan1Q(numCPW = "3",sparam="S21,",dcsweepch = "1")
    routine.cavitysearch()
    print("start step1\n")
    print(routine.cavity_list)
    print(routine.total_cavity_list)
    # for i in routine.total_cavity_list:
    for i in routine.total_cavity_list:
        print("start step2\n")
        routine.powerdepend(i)
        f_bare = mean(routine.cavity_list[str(i)])
        print("start step3\n")
        routine.fluxdepend(i,f_bare)
        print("start step4\n")
        routine.qubitsearch(i)
        break #test once
    # id = int(input("id? : "))
    # pyqum_path,task = jobid_search_pyqum(id)
    # amp_data,jobid  = pyqum_load_data(pyqum_path)