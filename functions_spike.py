def descripcion_inicial(x):
    display(x.head())
    display(x.shape)
    print('Valores nulos por variable:')
    display(x.isnull().sum())
    print('valores duplicados por variable:')
    display(x.info())
    display(x.describe().T)
    
def fix_fecha(x):
    try:
        x['mes'] = x.date.apply(lambda y: y.split('-')[1])
        x['año'] = x.date.apply(lambda y: y.split('-')[0])
        x.sort_values(by=['date'], ascending=True, inplace=True)
    except:
        x['mes'] = x.Periodo.apply(lambda y: y.split('-')[1])
        x['año'] = x.Periodo.apply(lambda y: y.split('-')[0])
        x.sort_values(by=['Periodo'], ascending=True, inplace=True)
        x['Periodo'][89] = x['Periodo'][89].replace('13','12')
        #banco_central.Periodo = banco_central.Periodo.dt.date
        
def intervalo_precipitaciones_anual(region, inicio, termino):
    check_region(region)
    check_date(inicio)
    check_date(termino)
    precipitaciones_intervalo = precipitaciones[(precipitaciones['date'] >= inicio) & (precipitaciones['date'] <= termino)][['date', region]]
    precipitaciones_intervalo['mes'] = precipitaciones_intervalo.date.apply(lambda y: y.split('-')[1])
    precipitaciones_intervalo['año'] = precipitaciones_intervalo.date.apply(lambda y: y.split('-')[0])
    precipitaciones_intervalo = precipitaciones_intervalo.rename(columns={region: 'Precipitaciones registradas (mm)'})

    return precipitaciones_intervalo

def check_date(x):
    z = 0
    for y in precipitaciones.date:
        if x == y:
            z = 1
        else:
            pass
    if z == 1:
        print(x + ' se encuentra en el intervalo de fechas')
    else:
        print('La fecha: ' + x + 'no está en el rango de fechas')
        print('inicio periodo: ' + str(precipitaciones.date.min()))
        print('termino periodo: ' + str(precipitaciones.date.max()))
        print('Intenta con otra fecha dentro del rango')
        
def check_region(x):
    if x not in precipitaciones.columns:
        for y in precipitaciones.columns[1:8]:
            print(y)
        x = input('Ingresa el nombre aqui:')
        x = str(x)
        x = x.lower()
        if 'metropolitana' in x:
            x = 'Metropolitana de Santiago'
        elif 'bernardo'in x:
            x = 'Libertador Gral Bernardo O Higgins'
        elif 'ohiggins' in x:
            x = 'Libertador Gral Bernardo O Higgins'
        elif "o'higgins" in x:
            x = 'Libertador Gral Bernardo O Higgins'
        elif 'araucania' or 'araucania' in x:
            x = 'La Araucania'
        elif 'araucanía' in x:
            x = 'La Araucania'
        elif 'rios' or 'ríos' in x:
            x = 'Los Rios'
        elif 'ríos' in x:
            x = 'Los Rios'
        else:
            print('Revisa la función e intenta cambiar la región a una correcta.')
    return x

def precipitaciones_mensuales_df(x):
    region=['Coquimbo', 'Valparaiso', 'Metropolitana de Santiago', 'Libertador Gral  Bernardo O Higgins', 'Maule', 'Biobio', 'La Araucania', 'Los Rios']
    precipitaciones_mensuales = x[region + ['mes']].melt('mes')
    precipitaciones_mensuales = precipitaciones_mensuales.rename({'variable':'Región', 'value':'Precipitaciones registradas'}, axis=1)
    return precipitaciones_mensuales

def precipitaciones_anuales_df(x):
    region=['Coquimbo', 'Valparaiso', 'Metropolitana de Santiago', 'Libertador Gral  Bernardo O Higgins', 'Maule', 'Biobio', 'La Araucania', 'Los Rios']
    precipitaciones_anuales = x[region + ['año']].melt('año')
    precipitaciones_anuales = precipitaciones_anuales.rename({'variable':'Región', 'value':'Precipitaciones registradas'}, axis=1)
    return precipitaciones_anuales

def precipitaciones_anuales_graficos():
    ax = sb.relplot(data = precipitaciones_coquimbo, x= 'año', y= 'Precipitaciones registradas (mm)', kind='line')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    display(ax.fig.suptitle('Precipitaciones anuales en '+region1, fontsize=15, fontdict={"weight": "bold"}))
    ax = sb.relplot(data = precipitaciones_metropolitana, x= 'año', y= 'Precipitaciones registradas (mm)', kind='line')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    display(ax.fig.suptitle('Precipitaciones anuales en '+region2, fontsize=15, fontdict={"weight": "bold"}))
    
def precipitaciones_mensuales_graficos():
    ax = sb.relplot(data = precipitaciones_coquimbo, x= 'mes', y= 'Precipitaciones registradas (mm)', kind='line')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    display(ax.fig.suptitle('Precipitaciones mensuales en '+region1, fontsize=15, fontdict={"weight": "bold"}))
    ax = sb.relplot(data = precipitaciones_metropolitana, x= 'mes', y= 'Precipitaciones registradas (mm)', kind='line')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    display(ax.fig.suptitle('Precipitaciones mensuales en '+region2, fontsize=15, fontdict={"weight": "bold"})) 

def precipitaciones_mensuales_maule():
    lista_años = ['1982','1992','2002','2012','2019']
    precipitaciones_intervalo_años = precipitaciones[precipitaciones['año'].isin(lista_años)]
    precipitaciones_intervalo_años = precipitaciones_intervalo_años[['mes', 'año', 'Maule']]
    precipitaciones_intervalo_años = precipitaciones_intervalo_años.pivot('mes', 'año', 'Maule')
    precipitaciones_intervalo_años = precipitaciones_intervalo_años.rename({'01':'Enero', '02':'Febrero', '03':'Marzo', '04':'Abril', '05':'Mayo', '06':'Junio', '07':'Julio', '08':'Agosto', '09':'Septiembre', '10':'Octubre', '11':'Noviembre', '12':'Diciembre'})
    ax = sb.lineplot(data=precipitaciones_intervalo_años)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)

def check_date_pib(x):
    z = 0

    if x in str(banco_central.Periodo):
        print(x + ' se encuentra en el intervalo de fechas')
    else:
        print('inicio periodo: ' + str(banco_central.Periodo.min()))
        print('termino periodo: ' + str(banco_central.Periodo.max()))
        print('Intenta con otra fecha dentro del rango')
def df_series_pib_func(inicio,termino,serie1, serie2):
    banco_central_pib = banco_central[[serie1, serie2, 'mes','año', 'Periodo']]
    df_series_pib = banco_central_pib
    df_series_pib = df_series_pib[(df_series_pib['Periodo'] >= pd.to_datetime(inicio).date()) & (df_series_pib['Periodo'] <= pd.to_datetime(termino).date())][['Periodo', 'PIB_Agropecuario_silvicola', 'PIB_Servicios_financieros']]
    return df_series_pib
def grafico_series_pib(x):
    g = sb.lineplot(x= 'Periodo', y='value', hue='variable', data=pd.melt(df_series_pib, ['Periodo']))
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    ylabels = [y for y in g.get_yticks()/1000000]
    g.set_yticklabels(ylabels)
def series_historicas_pib(df,inicio,termino, serie1, serie2):
    df_series_pib = df
    df_series_pib.PIB_Agropecuario_silvicola = df_series_pib.PIB_Agropecuario_silvicola.astype(float)
    df_series_pib.PIB_Servicios_financieros = df_series_pib.PIB_Servicios_financieros.astype(float)
    df_series_pib = df_series_pib.rename({'variable':'value', 'value':'PIB en millones'}, axis=1)
    grafico_series_pib(df_series_pib)    


def mes_numero(x):
    x = x.replace('Ene','01').replace('Feb','02').replace('Mar','03').replace('Abr','04').replace('May','05').replace('Jun','06').replace('Jul','07').replace('Ago','08').replace('Sep','09').replace('Oct',10).replace('Nov','11').replace('Dic','12')
    return x

def precio_leche_df(x):
    x = x.rename({'Anio':'año', 'Mes':'mes'}, axis=1)
    x = mes_numero(x)
    fecha = x['año'].astype(str) +'-'+ x['mes'].astype(str) +'-'+'01'
    x['date'] = pd.to_datetime(fecha).dt.date
    
    x = x.drop(['mes', 'año'], axis=1)
    return x
def trimestres(x):
    a = {'01':'1', '02':'1', '03':'1', '04':'2', '05':'2', '06':'2', '07':'3', '08':'3', '09':'3', '10':'4', '11':'4', '12':'4'}
    x = x.replace(a)
    return x
def combinar_dfs(df1,df2,df3):
    precipitaciones['date'] = pd.to_datetime(precipitaciones['date']).dt.date
    merge_df = pd.merge(df1, df2, on='date')
    try:
        df3 = df3.rename({'Periodo':'date'}, axis=1)
    except:
        pass
    merge_df = pd.merge(merge_df, df3, on='date')
    try:
        merge_df['Trimestre'] = trimestres(merge_df.mes_y)
        merge_df = merge_df.rename({'año_x':'año', 'mes_x':'mes'}, axis=1)
        del merge_df['mes_y']
        del merge_df['año_y']
    except:
        pass
    return merge_df

def fix_df(x):
    x1 = x[['año','mes','Trimestre','date']]
    del x['año']
    del x['mes']
    del x['Trimestre']
    del x['date']
    x = x.apply(pd.to_numeric, errors='coerce')
    x['date'] = x1['date']
    x['año'] = x1['año']
    x['mes'] = x1['mes']
    x['Trimestre'] = x1['Trimestre']
    return x

def subset_variables(x, variab):
    variables = [col for col in variab]
    y = x[['date','año','Trimestre','mes','Precio_leche']]
    x = x.reindex(variables, axis=1)
    x = x[variables]
    z = pd.merge(x, y, on='Precio_leche')
    return z
def analisis_corr():
    corr_df = df.corr()[['Precio_leche']].sort_values(by=['Precio_leche'])
    corr_neg_df = corr_df.head(15)
    corr_pos_df = corr_df.tail(15).sort_values(by=['Precio_leche'], ascending = False)
    trans_df = corr_pos_df[corr_pos_df['Precio_leche'] >= 0.5].T
    variables_final = [col for col in trans_df]
    print('Las variables que cumplen el criterio son:')
    for x in variables_final[1:]:
        print(x)
    print('Muestra del dataframe final:')

    df_final = subset_variables(df,variables_final)
    display(df_final.head(5))
    display(df_final.shape)
    return df_final

def corr_graph():
    corr_df = df.corr()[['Precio_leche']].sort_values(by=['Precio_leche'])
    corr_neg_df = corr_df.head(15)
    corr_pos_df = corr_df.tail(15).sort_values(by=['Precio_leche'], ascending = False)
    x = subset_variables(df,variables_final)
    variables = [col for col in x]
    x = x.reindex(variables, axis=1)
    x = x[variables]
    plot = df_final.corr()
    f, ax = plt.subplots(figsize=(10,10))
    print('Las 15 variables con mayor correlación negativa')
    display(corr_neg_df)
    print('Las 15 variables con mayor correlación Positiva')
    display(corr_pos_df)
    print('Matriz de correlación entre variables seleccionadas')
    sb.heatmap(plot, square=True, annot = True, annot_kws={'size':10}) 
def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2_square)
def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square












