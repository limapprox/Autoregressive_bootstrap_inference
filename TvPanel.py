# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:26:50 2022

@author: pavit
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import warnings
import seaborn as sns

# import geopandas as gpd
# import cbsodata

def clean_DFmort(df_mortRates, data_dateRange ):
    df_mortNew = df_mortRates.loc[df_mortRates['Stroomtype'] == 'Nieuwe contracten ']
    df_mortTotalMatur = df_mortNew.loc[df_mortNew['RenteVastPeriode'] == 'Totaal * ']
    x_mort = pd.DataFrame(df_mortTotalMatur['waarde'].values, index= df_mortTotalMatur['Periode '])
    x_mort.columns = ['Mortgage %']
    
    return x_mort[data_dateRange[0]:data_dateRange[1]]

def clean_Data(df_nvm,df_gemID,x_names,macro_names, ir10yrs,x_cpi,df_unemp,dateRange,df_IPI,freq,will2Buy):      
    NVMGems = np.unique(df_nvm['munid']) 
    nvm_gemCodes = NVMGems[~np.isnan(NVMGems)]
    df_NVMsorted = df_nvm.set_index( pd.to_datetime(df_nvm[['year', 'month', 'day']])).to_period(freq) 
    lY = []
    lX = []
    useGems = []
    count_transac = []
    counter = []
    for m,gemID in enumerate(nvm_gemCodes):
        dfNVM_i = df_NVMsorted[df_NVMsorted['munid'] == gemID]
        df = dfNVM_i.groupby(dfNVM_i.index).mean()[dateRange[0]:dateRange[1]]
        counter = dfNVM_i.groupby(dfNVM_i.index).count()[dateRange[0]:dateRange[1]]           
        count_transac.append(np.min(counter['price']))
        if count_transac[m] < 10:
            None
        else:
            useGems.append(gemID)
            df['size'] = np.log(df['size'])
            lX.append(df[x_names])
            lY.append(df['price'])
    
    useGems = np.asarray(useGems)
    # useGems[useGems==91.0] = 1900 #Sneek, Ysbrechtum etc changed to Súdwest-Fryslân
    useGems[useGems==612.0] = 1930 # Spijkenisse changed to Nissewaard
    cities = np.asarray([df_gemID[df_gemID['Gemcode2021']==gem].iloc[:,-1].\
                to_string(index=False) for gem in useGems])
    mY = np.asarray(lY).T
    
                        ##### CARTOGRAPHICS CODE####
    # data = pd.DataFrame(cbsodata.get_data('83765NED', select = ['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
    # data['Codering_3'] = data['Codering_3'].str.strip()
    # geodata_url = "https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json"
    # gemeentegrenzen = gpd.read_file(geodata_url)   
    # gemeentegrenzen["statcode"] = gemeentegrenzen["statcode"].str.lstrip('GM0').astype(float)
    # avgPriceMunic = pd.DataFrame(np.c_[useGems, np.std(mY,0)])
    # geoDataMunicip = pd.merge(gemeentegrenzen,avgPriceMunic ,
    #                         left_on = "statcode", right_on = avgPriceMunic.iloc[:,0])
    # # geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2021_gegeneraliseerd&outputFormat=json'
    # gemeentegrenzen = gpd.read_file(geodata_url) 
    # gemeentegrenzen = pd.merge(gemeentegrenzen, data,
    #                         left_on = "statcode", 
    #                         right_on = "Codering_3")
    # # sns.set_theme()
    # fig, ax = plt.subplots(figsize=(16, 16))
    # gemeentegrenzen.plot(ax = ax,column='GeboorteRelatief_25', 
    #                       edgecolor='black',color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372))
    # geoDataMunicip.plot(ax=ax, column = geoDataMunicip.iloc[:,-1], 
    #                         edgecolor='black', legend=True,legend_kwds={'shrink': 0.3})
    # _=ax.axis('off')
    # plt.title('Standard Deviation of Nominal House Prices', fontsize=25)
    # plt.savefig('STDHousePrice_Map.eps', bbox_inches='tight')
                                #### END ####
                                
    cities[cities == ' s-Gravenhage'] = 'The Hague' #International name for Den Haag
    mX_char = np.stack(lX)[:,1:,:] #(N x T x d) #if no counter change this to mX_char
    d = mX_char.shape[-1]
    # mY_real = np.log(np.multiply(1/(x_cpi.loc[dateRange[0]:dateRange[1]].values/100),mY) ) 
    mY_real_IDX = np.log(np.multiply(1/(x_cpi.loc[dateRange[0]:dateRange[1]].values),(mY/mY[0,:]*100))*100)[1:,:] #skip Base year since log(1) = 0
    mW = np.c_[will2Buy, df_unemp]
    mZ = np.ones((mX_char.shape[0],mX_char.shape[1],d+len(macro_names)+1)) 
    mZ[:,:,-mW.shape[-1]:] = mW   
    
    # mZ = np.ones((mX_char.shape[0],mX_char.shape[1],d + 1)) 
    
    mZ[:,:,1:d+1] = mX_char
    
    
    return mY_real_IDX, mZ[:,:,1:], mZ, cities, useGems
# -------------------------------------------------------------------------------------------
def estK(u,h):
    '''
    @Purpose
    ----------
    To estimate the kernel function using Epanechnikov's kernel.
    
    @Parameters
    ----------
    u : Argument for the kernel fucntion.
    h : The bandwidth used for the kernel estimation.

    @Returns
    -------
    mK : Diagonal matrix with the estimated weights on the diagonal.
    '''
    k_u = np.where(np.abs(u/h) <=1, 0.75*(1-(u/h)**2),0 )
    k_h = (k_u)/h
    mK = np.diagflat(k_h)   
    
    return mK

def est_mdl(mY, mX, t_i,T,N, mK):
    '''
    @Purpose
    ----------
    To obtain model estimates for gt, the local component Beta, Gamma and finally...
    ... the dummy variable by using the local linear dummy variable (LLDV) approach.
    
    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT).

    @Returns
    -------
    The estimated coefficients (Txd+1) and the estimated dummy variables (Nx1).
    '''
    X_gt = np.ones(T)
    d = mX.shape[-1] +1
    mX_bar = np.mean(mX,axis=0) 
    vY_bar = np.mean(mY,axis=1)
    
    THETA_hat = np.zeros((T,d*2))
    alpha_hat = np.zeros((N,T))
    for i in range(1, T+1): #change back to T+1
        t_i_min_t = t_i - i/T
        lZ_tild = []
        lY_tild = []
        lZ = []
        lZZ=[]
        lZY = []
        for j in range(N):
            vK = (np.diagonal(mK[i-1,:,:])).reshape(len(mK[i-1,:,:]),1)
            mZi = np.c_[X_gt,mX[j,:,:],t_i_min_t,mX[j,:,:]*(t_i_min_t.reshape(T,1)) ]
            lZ.append(mZi)
            mZ_bar = np.c_[X_gt,mX_bar,t_i_min_t,mX_bar*(t_i_min_t.reshape(T,1))]
            lZ_tild.append(  mK[i-1,:,:]**(1/2)@mZi- 1/np.sum(mK[i-1,:,:])*(vK**(1/2)@vK.T)@\
                          (mZi-mZ_bar) )
            lY_tild.append(  mK[i-1,:,:]**(1/2)@mY[:,j]- 1/np.sum(mK[i-1,:,:])*(vK**(1/2)@vK.T)@\
                                      (mY[:,j]-vY_bar) )
            lZZ.append(lZ_tild[j].T@lZ_tild[j])
            lZY.append(lZ_tild[j].T@lY_tild[j])
        THETA_hat[i-1,:] = np.linalg.inv(np.sum(lZZ,0))@ np.sum(lZY,0)
        alpha_hat[:,i-1] =  [1/np.sum(mK[i-1,:,:])*vK.T @\
        ((mY[:,y]-vY_bar)-(lZ[y]-mZ_bar)@THETA_hat[i-1,:]) for y in range(N)]

    return THETA_hat[:,:d], np.mean(alpha_hat,axis = 1)

def parLOUOCV(mY,mZ,mYmin_i, mXmin_i,t_i, mK_hi):
    '''
    @Purpose
    ----------
    This function is used to parallelize the inner loop in estBandwithLOUOCV(...)...
    ... in order to obtain the residuals.
    '''
    T,N = mYmin_i.shape
    theta_hat, _ = est_mdl(mYmin_i, mXmin_i, t_i, T, N, mK_hi)  
    return (mY - np.sum(np.multiply(theta_hat,mZ),1))


def estBandwithLOUOCV(mY,mX,mZ,t_i,T,N,saveName):
    '''
    @Purpose
    ----------
    To obtain the optimal bandwidth by using the leave-one-unit-out cross-validation...
    ... method see Sun, Y., R. J. Carroll, and D. Li (2009). Semiparametric estimation...
    ...of fixed-effects panel data varying coefficient models. 

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    saveName : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    h_opt : Optimal bandwidth according to the LOUOCV.
    '''
    step_size = 0.01
    h = np.arange(0.1, 0.6+step_size, step_size) #endpoint+step size to include endpoint
    lK = [np.asarray([estK(t_i- i/T, h_i) for i in range(1,T+1)]) for h_i in h]
    lWSEP = []
    Md = np.identity(N*T) - 1/T * np.kron(np.identity(N), np.ones((T,T)))
    for j in range(h.shape[0]):
        eps = np.asarray(Parallel(n_jobs=-1)(delayed(parLOUOCV)(mY[:,n],mZ[n,:,:],np.delete(mY,n ,1), np.delete(mX,n ,0),t_i, lK[j]) for n in range(N)))
        lWSEP.append( np.sum((Md@(np.asarray(eps).flatten('F')))**2))
    h_opt = h[np.argmin(lWSEP)]  
    
    sns.set_theme()
    plt.plot(h,lWSEP)
    plt.title('Leave-One-Unit-Out Cross-Validation',fontsize=17)
    plt.plot(h,lWSEP, markevery=[np.argmin(lWSEP)],ls="",marker="o")
    plt.savefig('h_'+str(np.round(h_opt,2))+'LOUOCV_'+saveName+'.eps')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    np.savetxt(str(h_opt)+"LOUOCV_"+saveName+".csv", lWSEP)
    
    return h_opt

def getFitPlots(mY,mZ,N,idx,mTheta_hat,vAlpha_hat,lIndivNames,saveName):
    '''
    @Purpose
    ----------
    Fit plots may contribute to a graphical analysis by assesing the model predicitons.
    
    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    mTheta_hat : The estimated coefficients (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    lIndivNames : List of the individual names (the name for each i in N) used for...
                    plot titles and file names.
    saveName : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files.
    '''
    mY_hat = np.asarray([vAlpha_hat[j] +\
            np.sum(np.multiply(mTheta_hat,mZ[j,:,:]),axis=1) for j in range(N)]).T
    for n in range(N):
        plt.scatter(idx, mY[:,n], label ='$y$', facecolor='none', edgecolor=f'C{n}',alpha=0.8)
        plt.plot(idx, mY_hat[:,n], label = '$\hat{y}$')
        plt.legend()
        plt.title(lIndivNames[n],fontsize=17)
        plt.savefig(saveName+'_Fit_'+str(lIndivNames[n])+'.eps')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()
    
def getYmeanPlots(mY,mZ,T,N,idx,mTheta_hat,saveFileNames,ytPredBase):
    '''
    @Purpose
    ----------
    May contribute to a graphical analysis.
    
    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    T : Amount of observations.
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    mTheta_hat : The estimated coefficients (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    saveFileNames : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files.
    '''
    sns.set_theme()
    ytBar_pred = np.asarray([np.mean(mZ,0)[t,:]@mTheta_hat[t,:] for t in range(T)])
    # fig = plt.figure(figsize =[15 ,10])
    # plt.subplot(1, 2, 1)
    # for n in range(N):
    #     plt.scatter(idx,mY[:,n], facecolor='none', edgecolor=f'C{n}',alpha=0.8)
    # plt.plot(idx,ytBar_pred ,'k--' ,linewidth=2,label=r'$\overline{y}_{t}^{pred}$')
    # plt.plot(idx,np.mean(mY,1),'k' ,linewidth=1.8,label=r'$\overline{y}_{t}$')
    # plt.legend(loc='lower left',fontsize='x-large')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.subplot(1, 2, 2)
    plt.plot(idx, ytPredBase,'b--' ,linewidth=2,label=r'$\overline{\hat{y}}_{t}^{base}$')    
    
    plt.plot(idx, ytBar_pred,'r--' ,linewidth=2,label=r'$\overline{\hat{y}}_{t}^{macro}$')
    plt.plot(idx,np.mean(mY,1),'k' ,linewidth=1.8,label=r'$\overline{y}_{t}$')
    plt.legend(loc='lower left',fontsize='medium')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    # plt.savefig('House_yMeanSubPlots_'+saveFileNames+'.eps')
    plt.savefig('YmeanPLots.eps')
    plt.show()

    
def step2__4AWB(mX,t_i,T,N,mK,gamma,mEpsilon_tild, first_term):
    '''
    @Purpose
    ----------
    Contains the repetitative steps from the AWB algorithm which is convenient...
    ... for parallelization.

    @Parameters
    ----------
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT), by using h.
    gamma : Controls for the dependence and heteroskedasticity in the...
        ... autoregressive part (step2).
    mEpsilon_tild : The oversmoothed residuals of size (TxN).
    first_term : The sum of the oversmoothed model excluding the error terms.

    @Returns
    -------
    mTheta_star : The simulated coefficients (Txd+1).
    '''
    ##step2##
    vVega = np.random.normal(0,(1-gamma**2)**(1/2),T)
    vGhi_star = np.full((T,1),np.random.standard_normal())
    for i in range(1,T):
        vGhi_star[i] = gamma*vGhi_star[i-1] + vVega[i]

    ##step3##
    mEpsilon_star = np.multiply(vGhi_star,mEpsilon_tild)
    mY_star = first_term + mEpsilon_star
    
    ##step4##
    mTheta_star,_ = est_mdl(mY_star, mX, t_i,T,N, mK)
    
    return mTheta_star

def getSimultAlpha(B,alpha,cntrd_stat):
    '''
    @Purpose
    ----------
    To estimate alpha (signficance level) for the simultaneous confidence bands...
    ... where this function coincides with step 2 of the three-step procedure.
    
    @Parameters
    ----------
    B : The amount of bootstrap simulations.
    alpha : The theoretical significance level.
    cntrd_stat : The centered bootstrap coefficient.

    @Returns
    -------
    The pointwise error which has the closest coverage to the theoretical coverage.
    '''
    alphas = np.linspace(1/B, alpha, B+1)
    f_prev = None
    for i,alpha_p in enumerate(alphas):
        cntrdB = np.sort(cntrd_stat, axis = 1)
        L = cntrdB[:,int(B*alpha_p/2)].reshape(cntrd_stat.shape[0],1)
        U = cntrdB[:,int(B*(1-alpha_p/2))].reshape(cntrd_stat.shape[0],1)   
        f = np.all(np.less_equal(L,cntrd_stat ) & np.less_equal(cntrd_stat,U),axis=0)
        covRat =  abs(sum(f)/B - (0.95))
        if f_prev is not None and f_prev<=covRat:
            break
        f_prev = covRat
        
    return alphas[i-1]

def estCI(alpha,T, B, t_i, idx, b_star, b_tild, b_hat, name, title_name,saveFileNames ):
    '''
    @Purpose
    ----------
    To save the centered bootstrap statistics (.csv) and plot and save the...
    ... estimated coefficients simultaneously with their pointwise and simultaneous...
    ... confidence bands.
    
    @Parameters
    ----------
    alpha : The significance level used to construct simultaneous bands.
    b_star : The bootstrapped coefficients (TxB).
    b_tild : The oversmoothed coefficients (Tx1).
    b_hat : The estimated coefficients (Tx1).
    T : Amount of observations.
    B : Amount of bootstrap simulations.
    t_i : Grid points over which one wants to estimate (Tx1).
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    name : Variable name used for saving the plots.
    title_name : Variable name used for the plot titles.
    saveFileNames : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files...
    ... some intermediate results are saved in csv files.

    '''
    cntrdB = np.sort((b_star - b_tild.reshape(T,1)), axis = 1)
    CI = [ b_hat-cntrdB[:,int(B*(1-alpha/2))], b_hat-cntrdB[:,int(B*alpha/2)] ] 
    alpha_theor = 0.05
    CIpw = [ b_hat-cntrdB[:,int(B*(1-alpha_theor/2))], b_hat-cntrdB[:,int(B*alpha_theor/2)] ]
    np.savetxt(name+'ST_PW_'+saveFileNames+'.csv', np.c_[b_hat,CI[0],CI[1],CIpw[0],CIpw[1]])
    
    sns.set_theme()
    plt.plot(idx,b_hat, label= name , color = 'k')
    plt.plot(idx,CI[0], '--', label= 'ST', color = 'r' )
    plt.plot(idx,CI[1], '--', color = 'r' )
    plt.plot(idx,CIpw[0], ':', label= 'PW', color = 'b' )
    plt.plot(idx,CIpw[1], ':', color = 'b' )
    
    plt.title(title_name, fontsize = 17)
    plt.savefig(str(name)+'_'+saveFileNames+'.eps')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
    
def estConfidenceBounds(mY,mX,mZ,t_i,T,N,idx,h,mK,mTheta_hat,vAlpha_hat,ctrl_names,var_titles,saveFileNames):
    '''
    @Purpose
    ----------
    This function uses the AWB procedure to obtain bootstrap critical values for... 
    ... constructing pointwise confidence intervals and simultaneous confidence bands.
    
    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         ... or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    h : The bandwidth that is used for the estimated model.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT), by using h.
    mTheta_hat : The estimated coefficients for all variables (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    ctrl_names : List containing the variable names used for saving the plots.
    var_titles : List containing the variable names used for the plot titles.
    saveFileNames : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files,...
    ... estimated coefficients and both type confidence bounds are saved for...
    ... each variable in .csv files.
    '''
    ##magic numbers##
    B = 1499 #999
    C_tild = 2
    h_tild = C_tild* h**(5/9)
    gamma = 0.2
    
    ##step1##
    mK_tild = np.asarray([estK(t_i- i/T, h_tild) for i in range(1,T+1)])
    mTheta_tild, vAlpha_tild = est_mdl(mY, mX, t_i,T,N, mK_tild)
    mEpsilon_tild = np.asarray([mY[:,j]- vAlpha_tild[j] -\
            np.sum(np.multiply(mTheta_tild,mZ[j,:,:]),axis=1) for j in range(N)]).T
    
    first_term = mY - mEpsilon_tild #step 3 
    # mTheta_star = np.asarray(
    #     [step2__4AWB(mX,t_i,T,N,mK,gamma,mEpsilon_tild, first_term) for _ in range(B)] )
    mTheta_stars = np.asarray(Parallel(n_jobs=-1)(delayed(step2__4AWB) \
                    (mX,t_i,T,N,mK,gamma,mEpsilon_tild, first_term) for _ in range(B)) )
    
    ##step5##
    alpha = 0.05
    for i in range(mTheta_stars.shape[-1]):
        cntrd_Bt = mTheta_stars[:,:,i].T- mTheta_tild[:,i].reshape(T,1)  
        alpha_s2 = getSimultAlpha(B,alpha,cntrd_Bt)
        estCI(alpha_s2,T, B, t_i, idx, mTheta_stars[:,:,i].T, mTheta_tild[:,i], \
              mTheta_hat[:,i], ctrl_names[i],var_titles[i],saveFileNames)
            
def getTimeVarPanelRes(mY,mX,mZ,idx,lIndivNames,saveFileNames,all_names,all_titles):
    '''
    @Purpose
    ----------
    To obtain plots of estimation results. 
    
    @Parameters
    ----------
    mY : Dependent variable of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension excluding time-varying intercept gt (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         ... or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    lIndivNames : List of the individual names (the name for each i in N) used for...
                    plot titles and file names.
    saveFileNames : The file name for saving the plots (.eps) and estimates (.csv).
    all_names : List containing the variable names used for saving the plots.
    all_titles : List containing the variable names used for the plot titles.

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files...
    ... intermediate results are saved in .csv files.

    '''
    T,N = mY.shape
    t_i = np.arange(1,T+1)/T
    ##choose which BandWidth method you want to use##
    h_opt = estBandwithLOUOCV(mY,mX,mZ,t_i,T,N,saveFileNames)
    # h_opt = 0.42
    print('Optimal bandwidth: h = ',h_opt)
    mK_opt = np.asarray([estK(t_i- i/T, h_opt) for i in range(1,T+1)])
    mTheta_hat, vAlpha_hat = est_mdl(mY,mX,t_i,T,N,mK_opt)
    """ INFERENCE FOR THE EXAMPLE CODE:
    mXbase = mX[:,:,:-3]
    mZbase = mZ[:,:,:-3]
    mTheta_hatB, vAlpha_hatB = est_mdl(mY,mXbase,t_i,T,N,mK_opt)
    ytBar_predBase = np.asarray([np.mean(mZbase,0)[t,:]@mTheta_hatB[t,:] for t in range(T)])
    #####
    ##Fit plots for each individual and saves them##
    getFitPlots(mY,mZ,N,idx,mTheta_hat,vAlpha_hat,lIndivNames,saveFileNames)
    ##Cross-sectional mean plots##
    getYmeanPlots(mY,mZ,T,N,idx,mTheta_hat,saveFileNames,ytBar_predBase)
    """
    estConfidenceBounds(mY,mX,mZ,t_i,T,N,idx,h_opt,mK_opt,mTheta_hat,vAlpha_hat,all_names,all_titles,saveFileNames)
    
    
def main():  
    """EXAMPLE INPUT PACKAGE
    np.random.seed(1234) #comment/remove for randomness
    warnings.filterwarnings("ignore") #supresses warnings in the console 
    data_dateRange = ['2006-01-01', '2020-12-01'] #[startYYYY-MM-DD, endYYYY-MM-DD]
    pre_sampleDate = '2005-12-01' #specify if needed for data transformations
    freq = "M" #specify desired frequency: 'D', 'Q', 'Y', more options at https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    
    ####Read data files####
    df_gemID = pd.read_csv('gem2021.csv',sep=';')
    # df_CPI = pd.read_csv('inflatie.csv', sep = ';',parse_dates=['Perioden'], index_col=['Perioden']).loc[pre_sampleDate:data_dateRange[1]]
    df_CPI_2005 = pd.read_csv('CPI_BaseYear2005-12.csv', parse_dates = ['DATE'], index_col= ['DATE'])
    
    df_mortRates = pd.read_csv('MortageRates_DNB.csv', parse_dates=['Periode '] )
    df_IPI = pd.read_csv('IPI_(GDP_proxy).csv', sep=';',parse_dates=['Periods'], index_col=['Periods'])[data_dateRange[0]:data_dateRange[1]]
    # df_AEX = pd.read_csv('AEX_2003_2022.csv', usecols=['Date', 'Close'], parse_dates=['Date'],index_col = ['Date'])[data_dateRange[0]:data_dateRange[1]]
    df_unempl = pd.read_csv('unemp_NL_2003tm2021.csv', sep = ';',parse_dates=['Perioden'], index_col=['Perioden'])[data_dateRange[0]:data_dateRange[1]]
    df_consSurv = pd.read_csv('ConsumerSentiment_1986-2022.csv',sep=';', index_col=['Periods'],parse_dates=['Periods'])[data_dateRange[0]:data_dateRange[1]] 
    w_mort = clean_DFmort(df_mortRates, data_dateRange)   
    IPI = df_IPI['Production_1']
    idx = IPI.index.to_period().to_timestamp()
    will2Buy = df_consSurv['WillingnessToBuy_3'].values
    
    # ir = pd.read_csv('IR_LT10yrs_NL.csv',index_col=['DATE'])[data_dateRange[0]:data_dateRange[1]] 
    
    ####empirical study####
    macro_var = ['Willingness to Buy'] + ['Unemployment%']
    nvm_variables = ['rooms','nfloors', 'size' ] + ['parking','garden', \
              'constr19451959', 'constr19601970', 'constr19711980',\
                  'constr19811990','constr19912000', 'constrgt2000' ]    
    useColsNVM = ['price','munid','apartment', 'year', 'month', 'day']
    df_NVM = pd.read_stata('NVM_cleaned_040522.dta',columns=useColsNVM+nvm_variables)
    mY, mX, mZ,cities, GemCodes  = clean_Data(df_NVM[df_NVM['apartment']==0], df_gemID, nvm_variables,
                    macro_var, w_mort, df_CPI_2005,df_unempl,[pre_sampleDate,data_dateRange[1]],IPI,freq,will2Buy)        
    

    ##Naming the variables for labelling, plot titles and file names.##
    var_name = [ '# Rooms' , '# Floors', 'Size' ,  'Parking%', 'Gardens%',\
      '%ConstYear 45-59','%ConstYear 60-70','%ConstYear 71-80','%ConstYear 81-90',
      '%ConstYear 91-00', '%ConstYear 00-20']
    var_titles = ['Avg. Number of Rooms', 'Avg. Number of Floors', 'Log Avg. Size', \
      'Percentage of Parking Space Availability', 'Percentage of Gardens Present', \
      'Percentage Constructed in Year: 1945-1959','Percentage Constructed in Year: 1960-1970', \
      'Percentage Constructed in Year: 1971-1980','Percentage Constructed in Year: 1981-1990', \
      'Percentage Constructed in Year: 1991-2000', 'Percentage Constructed in Year: 2000-2020']
    ##Make sure all variable names are lists [ ] before adding them.##
    all_names =  ['gt']+ var_name + macro_var
    all_titles = ['$\hat{g}_{t}$'] + var_titles +macro_var
    """
#-----------------------------------------------------------------------------------------------------------------
    """PACKAGE STARTS HERE"""
    ##Use mY of size (T x N), mX ~(N x T x d) and mZ ~(N x T x d+1).##
    saveFileNames = 'MacroUNempWill2' #specify name to save the plots and csv's   
    getTimeVarPanelRes(mY,mX,mZ,idx,cities,saveFileNames,all_names,all_titles)


   
    # return locals()
############### now call the main ############    
start = timeit.default_timer()
if __name__ == "__main__":
    main()
    # locals().update(main()) 
stop = timeit.default_timer()
print('\nTime: ', stop - start)  