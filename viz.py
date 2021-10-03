import pandas as pd
import numpy as np
import plotly.express as ex
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,mean_absolute_error,mean_squared_error
import pydeck as pdk
import math
#import altair as alt
import warnings
import pickle


st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def zillow():
    return pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/main_df.csv")

zillow_detail_df = zillow()

st.sidebar.header('Hello There! :wave:')
st.sidebar.subheader("""Welcome !""")

menubar = st.sidebar.radio("",['Home','Overview','Exploratory Data Analysis','House Price Predictions','About Us'])

if menubar == 'Home':
    st.title("Real Estate Analysis - California Bay Area :house:")


    with st.echo():

        st.write("Hello")
        st.text("This is text")
        st.header("Header")
        st.markdown("**Markdown**")
        st.caption('This is a string that explains something above.')
        st.title("Title")

#=========================================================================================================================================
# --------------------------------------------------Map ploting---------------------------------------------------------------------------#
#=========================================================================================================================================

if menubar == "Exploratory Data Analysis":

    st.dataframe(zillow_detail_df,width = 1000,height = 350)

    st.title("Geo Representation of Houses on Map")

    data = zillow_detail_df
    expander = st.expander("Fiter Map")

    with expander:
        st.subheader("Select the parameters below as per requirement")
        st.caption("Note: Please uncheck **Full Data** to update the maps based on selected parameters")
        r1,r2,r3 = st.columns((1,1,1))
        with r1:
            bedrooms = st.multiselect("Select the no. of bedrooms",zillow_detail_df.bedrooms.unique(),zillow_detail_df.bedrooms.unique())
        with r2:
            bathrooms = st.multiselect("Select the no. of bathrooms",zillow_detail_df.bathrooms.unique(),zillow_detail_df.bathrooms.unique())
        with r3:
            Hometype = st.multiselect("Select the Angle of View", zillow_detail_df.homeType.unique(),zillow_detail_df.homeType.unique())
        # with r4:
        #     city = st.selectbox("Select the cities",sorted(zillow_detail_df.city.unique()))
        data = zillow_detail_df[(zillow_detail_df.homeType.isin(Hometype))&((zillow_detail_df.bedrooms.isin(bedrooms)) & (zillow_detail_df.bathrooms.isin(bathrooms)))]


    if st.checkbox("Full Data"):
        data = zillow_detail_df


    mapspace1,map1,mapspace2,map2,mapspace3 = st.columns((0.05,1,0.05,1,0.05))

    with map1:

        fig = ex.scatter_mapbox(data[(data.notnull()['latitude'] & data.notnull()['price'])],
                                lat="latitude", lon="longitude", hover_name="city",
                                color = 'price', zoom=9, height=600,width = 650,size_max=12,size='price',hover_data = ['bedrooms','bathrooms','homeType'])
        fig.update_layout(mapbox_style="open-street-map")

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        st.plotly_chart(fig)

        st.text('''Description of this map




    Ends here''')

    with map2:
        st.pydeck_chart(pdk.Deck(initial_view_state={
                    "latitude": 37.629,
                    "longitude": -122.171,
                    "zoom": 9,
                    "pitch": 50,
                },
                layers=[
                    pdk.Layer(
                        "ColumnLayer",
                        data=data[['latitude','longitude','price','city','bedrooms','bathrooms','homeType']],
                        get_position=["longitude", "latitude"],
                        radius=90,
                        elevation_scale=25,
                        get_elevation = "price / 35000" ,
                        elevation_range=[0,25],
                        get_fill_color=["price /30000" , 90, 0],
                        pickable=True,
                        extruded=True,
                        auto_highlight=True,
                    ),
                ],tooltip = {
           "html": "<b>Price:</b> {price} <br/> <b>City:</b> {city} <br/> <b>Bedrooms:</b> {bedrooms} <br/> <b>Bathrooms:</b> {bathrooms} <br/> <b>HomeType:</b> {homeType} ",
           "style": {
                "backgroundColor": "steelblue",
                "color": "white"
           }
        }
            ))
        # midpoint = (np.average(data['latitude']),np.average(data['longitude']))
        # r = map(data, midpoint[0], midpoint[1], 8.5)
        # r.to_html("grid_layer.html")
        # HtmlFile = open("grid_layer.html", 'r', encoding='utf-8')
        # source_code = HtmlFile.read()
        # components.html(source_code,width = 650,height=600,scrolling = True)
        st.text('''Description of this map




Ends here''')

    st.write("---")
    c1,c2 = st.columns((1,6))

    arg = "Crime"
    height = 600
    with c1:
        st.write("""

    Options:

        """)
        arg = st.radio("",['city','Livability','Crime','Employment','Schools','Housing'])
        if arg =='city':
            height = 1500
        if arg == 'Livability':
            height = 800
        st.write("")
        agg = st.checkbox("NonAggregated")


    with c2:

        if agg:
            fig = ex.scatter(data_frame=zillow_detail_df,y=arg,x='price',size='price',size_max=15,
            color=arg,height=height,width = 1000,title = f"Scatter plot representation of Price vs {arg}",
            hover_data=['bathrooms','bedrooms','price','city'])
            st.plotly_chart(fig)
        else:
            df = zillow_detail_df[['price',arg]].groupby(by=arg).mean().reset_index()
            fig = ex.scatter(data_frame=df,y=arg,x='price',size='price',size_max=20,
            color=arg,height=height,width = 1000,title = f"Scatter plot representation of Price vs {arg}",
            hover_data=['price',arg])
            st.plotly_chart(fig)

#=======================================================================================================================================================
#--------------------------------------------------------------Time Series Data ------------------------------------------------------------------------
#=======================================================================================================================================================


@st.cache(allow_output_mutation=True)
def timeseriesdata():
    timeseries = pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/AllState.csv")
    return timeseries

@st.cache(allow_output_mutation=True)
def californiatimeseries():
    caltimeseries = pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/CaState.csv")
    return caltimeseries

# @st.cache
# def mortgagestimesries():
#     mort = pd.read_excel("C:\\Users\\STSC\\OneDrive - horizon.csueastbay.edu\\Desktop\\Ban 612\\Project\\Z_Data\\Timeseries data\\Mortgages.xlsx",
#          header=6,sheet_name = 'Full History')
#     mort = mort.iloc[:,[0,1,3,5]]
#     mort.columns = ['Date','30YearFRM','15YearFRM','5YearFRM']
#     return mort



timeseries = timeseriesdata()
caltimeseries = californiatimeseries()
# mortgagests = mortgagestimesries()

citiests = caltimeseries[caltimeseries.City.isin(zillow_detail_df.city.values)].groupby(by=['City','Year']).mean().reset_index()
counties = caltimeseries[caltimeseries.City.isin(zillow_detail_df.city.values)].groupby(by=['CountyName','Year']).mean().reset_index()


if menubar == "Overview":
    st.subheader("TimeSeries Chart of Median House Price In Various States of USA")
    st.write("")
    time1,time2,time3 = st.columns((1,0.05,0.35))

    with time1:
        expander2 = st.expander('Select States')
        s = ['NY','CA','NJ','HI','DC']
        with expander2:
            st.caption("Please select the states from the dropdown")
            s = st.multiselect("",timeseries.State.unique(),timeseries.State.unique()[:10])
        fig = ex.line(data_frame=timeseries[timeseries.State.isin(s)],x="Year",y='MedianPrice',color='State')
        fig.update_layout(width = 950,height =500)
        #plot_bgcolor = 'rgb(0,0,0)'
        fig.add_hline(y=timeseries[timeseries.State.isin(s)].MedianPrice.mean(), line_dash="dot",
          annotation_text="Median Price of all States",
          annotation_position="bottom right")
        fig.add_vrect(x0="2008", x1="2009.75",
          annotation_text="Economic Recision", annotation_position="top left",
          fillcolor="black", opacity=0.25, line_width=0)
        fig.add_vrect(x0="2020.1", x1="2021.25",
          annotation_text="Covid", annotation_position="top left",
          fillcolor="black", opacity=0.25, line_width=0)
        st.plotly_chart(fig)

    with time3:
        st.write("---")
        st.text("""


        Description for the chart goes here



Ends Here""")

    st.subheader("TimeSeries Chart of Median House Price In Various counties/cities of California")

    time4,time5,time6 = st.columns((0.35,0.05,1))


    with time4:
        c = st.radio("",['Counties','cities'])

        st.write("---")
        st.text("""Description for the chart goes here



ends Here""")

    with time6:

        if c == 'Counties':
            expander3 = st.expander('Select Counties')
            with expander3:
                st.caption("Please select the Counties from the dropdown")
                county = st.multiselect("",counties.CountyName.unique(),counties.CountyName.unique())
            fig = ex.line(data_frame=counties[counties.CountyName.isin(county)],x='Year',y='MedianPrice',color='CountyName')
            fig.add_hline(y=counties[counties.CountyName.isin(county)].MedianPrice.mean(), line_dash="dot",
            annotation_text="Median Price of all Counties",
            annotation_position="bottom right")
            fig.add_vrect(x0="2008", x1="2009.75",
            annotation_text="Economic Recision", annotation_position="top left",
            fillcolor="black", opacity=0.25, line_width=0)
            fig.add_vrect(x0="2020.1", x1="2021.25",
            annotation_text="Covid", annotation_position="top left",
            fillcolor="black", opacity=0.25, line_width=0)
            fig.update_layout(width = 950,height = 500)
            st.plotly_chart(fig)

        if c == 'cities':
            expander4 = st.expander('Select Cities')
            with expander4:
                st.caption("Please select the cities from the dropdown")
                #['Dublin','Danville','Alamo','San Francisco','Los Altos','Sunnyvale','Fremont']
                city = st.multiselect("",citiests.City.unique(),citiests.City.unique()[:10])
            fig = ex.line(data_frame = citiests[citiests.City.isin(city)],
            x='Year',y='MedianPrice',color = 'City')
            fig.add_hline(y=citiests[citiests.City.isin(city)].MedianPrice.mean(), line_dash="dot",
              annotation_text="Median Price of all Cities",
              annotation_position="bottom right")
            fig.add_vrect(x0="2008", x1="2009.75",
              annotation_text="Economic Recision", annotation_position="top left",
              fillcolor="black", opacity=0.25, line_width=0)
            fig.add_vrect(x0="2020.1", x1="2021.25",
              annotation_text="Covid", annotation_position="top left",
              fillcolor="black", opacity=0.25, line_width=0)
            fig.update_layout(width = 950,height = 500)
            st.plotly_chart(fig)


    # fig = ex.line(data_frame=mortgagests,x='Date',y=['30YearFRM','15YearFRM','5YearFRM'],labels={'Date':'Time','value':'Mortgage Rates'})
    # fig.update_layout(width = 800,height = 500)
    # st.plotly_chart(fig)


#=========================================================================================================================================================
#----------------------------------------------------------House Price Predictions -----------------------------------------------------------------------
#=========================================================================================================================================================
if menubar == "House Price Predictions":

    pipeline1 = pickle.load(open('housemodel.pkl','rb'))

    zillow_detail_df = zillow_detail_df.set_index('zpid')
    regression_df = zillow_detail_df

    regression_df = regression_df[['city','price','bathrooms','bedrooms','livingArea',
                            'homeType','taxAssessedValue','lotAreaValue','CalendarYear built',
                            'Price Square FeetPrice/sqft','HasFlooring','HasHeating','HasCooling','GarageSpaces','HasLaundary',
                             'FirePlaces','HasPool','HasSecurity','Stories','Livability','Crime','Employment','Housing','Schools']]

    reg_df = regression_df.drop('price',axis=1)

    ratings_dict ={'F':1,'D-':2,'D':3,'D+':4,'C-':5,'C':6,'C+':7,
                                 'B-':8,'B':9,'B+':10,'A-':11,'A':12,'A+':13}
    city_avgdf = zillow_detail_df[['city','price']].groupby(by = ['city']).mean().reset_index().sort_values(by = ['price'])
    city_avgdf['ranks'] = city_avgdf['price'].rank()
    city_dict = dict(zip(city_avgdf.city.values,city_avgdf.ranks.values.astype(np.int64)))

    def format(df):
        df = df[reg_df.columns]
        df.Schools = df['Schools'].map(ratings_dict)
        df.Crime = df['Crime'].map(ratings_dict)
        df.Employment = df['Employment'].map(ratings_dict)
        df.Housing = df['Housing'].map(ratings_dict)
        df.city = df['city'].map(city_dict)

        return df

    test_data = pd.read_csv('https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/testing.csv')


    n = st.text_input("Number of houses to predict",1)
    st.caption("choose a number between 1 and 10")
    for j in range(int(n)):
        st.write(f"Test Data {j+1}")
        i = np.random.randint(10,len(test_data)-10)
        one = (zillow_detail_df.loc[test_data.zpid,:]).iloc[i:i+1,:]
        st.write(one)

        but1 = st.button("Predict house price for test data with user model",key=j+100)

        if but1:

            predicted_price = pipeline1.predict(format(one))

            st.write("")
            met1,met2,met3,met4 = st.columns((1,1,1,1))
            with met1:
                st.metric('Listed Price',"$"+str(float(one.price.iloc[0])))
            with met2:
                st.metric(label="Predicted Value(user model)", value="$"+str(float(predicted_price[0])), delta=str(round((float(predicted_price[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))+"%")
                if (float(predicted_price[0])-float(one.price.iloc[0]))<0:
                    st.markdown(f"The house price predicted using XGBRegressor model is **{abs(round((float(predicted_price[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))}% less** than the original price listed on zillow")
                elif (float(predicted_price[0])-float(one.price.iloc[0]))>=0:
                    st.markdown(f"The house price predicted using XGBRegressor model is **{abs(round((float(predicted_price[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))}% more** than the original price listed on zillow")
            with met3:
                st.metric(label="Zestimate Value(zillow model)", value="$"+str(float(one.zestimate.iloc[0])), delta=str(round((float(one.zestimate.iloc[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))+"%")
                if ((float(one.zestimate.iloc[0])-float(one.price.iloc[0])))<0:
                    st.markdown(f"The house price predicted using Zillow model(Zestimate Value) is **{abs(round((float(one.zestimate.iloc[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))}% less** than the original price listed on zillow")
                elif ((float(one.zestimate.iloc[0])-float(one.price.iloc[0])))>=0:
                    st.markdown(f"The house price predicted using Zillow model(Zestimate Value) is **{abs(round((float(one.zestimate.iloc[0])-float(one.price.iloc[0]))*100/float(one.price.iloc[0]),2))}% more** than the original price listed on zillow")

    st.write("")
    st.write("")
    st.write("---")


    userinputs = st.expander("Predict house prices with your own customized data")

    with userinputs:
        st.write("---")


        i1,i2,i3,i4,i5 = st.columns((1,1,1,1,1))
        i6,i7,i8,i9,i10 = st.columns((1,1,1,1,1))
        i11,i12,i13,i14,i15 = st.columns((1,1,1,1,1))
        i16,i17,i18,i19 = st.columns((1,1,1,1))
        i20,i21,i22,i23 = st.columns((1,1,1,1))

        input_list = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23]

        d = {}
        for j,i in enumerate(reg_df.columns):
            if reg_df[i].dtype == 'object':
                with input_list[j]:
                    v = st.selectbox(i.upper(),sorted(list(reg_df[i].unique())),key = i)
                    # v = ratings_dict.get(v,v)
                    # v = city_dict.get(v,v)
                    # if i =='city':
                    #     v = int(v)

            elif reg_df[i].dtype == 'int64':
                with input_list[j]:
                    if i == 'Livability':
                        v=st.slider(i.upper(),0,100,int(np.mean(reg_df[i])),key = i)
                    elif i == 'CalendarYear built':
                        v=st.slider(i.upper(),int(min(reg_df[i])),int(max(reg_df[i])),2021,key = i)
                    else:
                        v=st.slider(i.upper(),0,int(max(reg_df[i])),int(np.mean(reg_df[i])),key = i)
            else:
                with input_list[j]:
                    v= st.text_input(f"Enter Value for {i}",int(np.mean(reg_df[i])))
                    v = np.float(v)

            d.update({i:v})

        st.write("---",width = 700,height = 200)

        input = pd.DataFrame(d,index=[0])
        st.caption("User input values")
        st.dataframe(input)

        predict_button = st.button('Predict Price')

        if predict_button:
            st.write('The predicted house price with specifications cost approximateley $',
            np.round(pipeline1.predict(format(input)))[0])


if menubar == "About Us":

    pass
