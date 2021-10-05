import pandas as pd
import numpy as np
import plotly.express as ex
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
import pydeck as pdk
import math
#import altair as alt
import warnings
import pickle


st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def zillow():
    return pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/main_df.csv")

@st.cache(allow_output_mutation=True)
def ph():
    return pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/PriceHistory.csv")

@st.cache(allow_output_mutation=True)
def timeseriesdata():
    timeseries = pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/AllState.csv")
    return timeseries

@st.cache(allow_output_mutation=True)
def californiatimeseries():
    caltimeseries = pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/CaState.csv")
    return caltimeseries

timeseries = timeseriesdata()
caltimeseries = californiatimeseries()
# mortgagests = mortgagestimesries()

zillow_detail_df = zillow()
PriceHistory = ph()

df_joined = pd.merge(PriceHistory,zillow_detail_df, how='inner',left_on='Zpid',right_on='zpid')

st.sidebar.header('Hello There! :wave:')
st.sidebar.subheader("""Welcome !""")

citiests = caltimeseries.groupby(by=['City','Year']).mean().reset_index()
counties = caltimeseries.groupby(by=['CountyName','Year']).mean().reset_index()
menubar = st.sidebar.radio("",['Home','Overview','Exploratory Data Analysis','House Price Predictions','About Us'])

if menubar == 'Home':
    st.title("Real Estate Analysis - California Bay Area :house:")
    st.write("")

    st.markdown('''Real Estate is part and parcel of every ones life.One has to involve in process of buying the house at certain point of time in life.
From Real estate investor point of view , real estate investment analysis is basically the process of analyzing investment opportunities
to decide whether or not they’ll give you the profits you’re aiming for to achieve your investment goals and this is perhaps the most crucial part to success.
''')

    st.markdown('''     For normal induvidual house is a necessity and one of the big investments in their life.SO ,There are many factors which affect the house prices such as number of bedrooms, bathrooms, mortgage rates ,crime rates in neighborhood, schools in neighborhood and many other factors.
    Having a good overview on the market will help in buying or selling the house in the market at right price.''')

    st.write("")
    st.subheader("TimeSeries Chart of Average House Price In Various counties/cities of California")

    time14,time15,time16 = st.columns((0.35,0.05,1))


    with time14:
        c = st.radio("",['Counties','cities'])
        st.write("")
        st.markdown('''California is on of the hottest cities for real estate and growing strong day by day.
We have choose Bay Area as our reaserch area to analyse real estate market.
Below plot shows Average price for each of the counties and cities in california.
Grey Shaded Areas represent Economic recision and covid-19 pandemic respectively.

Various Cities and counties can be analysed by filtering options available at top and side expander bar

**Note: San Francisco county average price is declining since lockdown**''')


    with time16:

        if c == 'Counties':
            expander3 = st.expander('Select Counties')
            with expander3:
                st.caption("Please select the Counties from the dropdown")
                county = st.multiselect("",counties.CountyName.unique(),counties.CountyName.unique()[:10])
            fig = ex.line(data_frame=counties[counties.CountyName.isin(county)],x='Year',y='MedianPrice',color='CountyName')
            fig.update_layout(yaxis = dict(title = "Average Price"))
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
            fig.update_layout(yaxis = dict(title = "Average Price"))
            fig.add_vrect(x0="2008", x1="2009.75",
              annotation_text="Economic Recision", annotation_position="top left",
              fillcolor="black", opacity=0.25, line_width=0)
            fig.add_vrect(x0="2020.1", x1="2021.25",
              annotation_text="Covid", annotation_position="top left",
              fillcolor="black", opacity=0.25, line_width=0)
            fig.update_layout(width = 950,height = 500)
            st.plotly_chart(fig)

    st.subheader("TimeSeries Chart of Average House Price In Various States of USA")
    st.write("")
    time1,time2,time3 = st.columns((1,0.05,0.35))

    with time1:
        expander2 = st.expander('Select States')
        s = ['NY','CA','NJ','HI','DC']
        with expander2:
            st.caption("Please select the states from the dropdown")
            s = st.multiselect("",timeseries.State.unique(),timeseries.State.unique()[:10])
        fig = ex.line(data_frame=timeseries[timeseries.State.isin(s)],x="Year",y='MedianPrice',color='State')
        fig.update_layout(width = 950,height =500,yaxis = dict(title = "Average Price"))
        #plot_bgcolor = 'rgb(0,0,0)'
        fig.add_vrect(x0="2008", x1="2009.75",
          annotation_text="Economic Recision", annotation_position="top left",
          fillcolor="black", opacity=0.25, line_width=0)
        fig.add_vrect(x0="2020.1", x1="2021.25",
          annotation_text="Covid", annotation_position="top left",
          fillcolor="black", opacity=0.25, line_width=0)
        st.plotly_chart(fig)

    with time3:
        st.write("")
        st.write("")
        st.markdown("""This chart show all the average house prices of all states across USA.California,DC,Hawai are top in average house prices.
Various states can selected from the expander bar on top of the chart.
Grey Shaded Areas represent Economic recision and covid-19 pandemic respectively.""")


    st.markdown('''So in our project we are trying to analyse which factors effect the house prices,which home types are available on market,which areas are having more price change etc.
We have choose to scrape data from zillow and analysed various analysis questions.A regression model is built to predict the house prices which takes user inputs like number of bedrooms,bathrooms,price/sqft,liviing area etc.. and predict the price.
''')

#=========================================================================================================================================
# --------------------------------------------------Map ploting---------------------------------------------------------------------------#
#=========================================================================================================================================

if menubar == "Exploratory Data Analysis":



    st.title("Analysis Questions")
    st.write("")


    question1 = st.expander("What type of homes are available in the market the most?")
    question2 = st.expander("What type of homes are most expensive and cheapest in terms of avg price/sqft?")
    question3 = st.expander("What is the average housing price for 3 bedrooms and 2 bathrooms in the bay area?")
    question4 = st.expander("What is the average housing price for 3 bedrooms and 2 bathrooms in the bay area by home type?")
    question5 = st.expander("What is the average housing price for 3 bedrooms and 2 bathrooms in the bay area by cities?")
    question6 = st.expander("What are the cities with oldest and newest buildings ready to be sold on zillow ?")
    question7 = st.expander("Which city in the Bay Area has the highest appreciation rate?")
    question8 = st.expander("Does CrimeRates,School rating and employment opportunities will impact the housing price?")
    question9 = st.expander("Which city has the lowest crime rate (safest), highest school rating and highest employment opportunities?")
    question10 = st.expander("Which city in the East Bay has the highest potential to be another real estate hot spot?")
    question12 = st.expander("How did housing prices change before the lockdown (2years) and after the lockdown?")


    with question1:
        st.write("---")
        q1_1,q1_2,q1_3 = st.columns((1,0.1,0.9))
        with q1_1:

            home_type = pd.DataFrame(zillow_detail_df['homeType'].value_counts()).reset_index()
            home_type.columns = ['HomeType','Count']
            fig = ex.pie(data_frame=home_type,names = 'HomeType',values = 'Count',labels={'Count':'No of Houses'},color = 'HomeType')
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            fig.update_layout(width = 650,height = 500,title = "Pie Chart for Different Hometypes",title_x = 0.5)
            st.plotly_chart(fig)

        with q1_3:
            st.write('')
            st.write("")
            st.markdown('''## Work:hammer::
- Create a series from the home type column of main_df.csv to show the frequency of each home type
- Plot a pie chart showing the **home type contribution**.

## Analysis:bulb::
- With **55.8%, single family** is the most popular home type in the Bay Area on Zillow.
- Following are condo and townhouse which contribute 25% and 11.3% of home type, respectively
- Multi family is pretty minor with 5.3%
- **Mobile Manufactured** is the less popular home type on Zillow with **2.5%**. This is predictable because the Bay Area one of the most expensive regions of the state''')


    with question2:
        st.write('---')
        q2_1,q2_2,q2_3 = st.columns((1,0.1,0.9))
        with q2_1:
            avg_price = zillow_detail_df[['homeType','Price Square FeetPrice/sqft']].groupby(['homeType']).mean().reset_index()
            avg_price.columns = ['homeType', 'Avg Price/sqft']
            avg_price = avg_price.sort_values(by = ['Avg Price/sqft'],ascending = False)
            fig = ex.bar(data_frame=avg_price,x='homeType',y='Avg Price/sqft',hover_data=['homeType','Avg Price/sqft'],color ='Avg Price/sqft',
                title = 'Average Price/Sqft For Each HomeType')
            fig.update_layout(width = 650,height = 500,title_x = 0.5)
            st.plotly_chart(fig)
        with q2_3:
            st.markdown('''## Work:hammer::
- Convert the Price Square FeetPrice/sqft column of main dataframe to numeric
- Get a dataframe of home type and price/sqft and group them by home type. Using DataFrame.mean() to calculate average price/sqft for each home type
- Plot a bar chart showing the Average Price/sqft for each home type.
- Rotate the text in x-asis to avoid too compacted text and easier to read

## Analysis:bulb::
- Single Family_ the most popular home type on Zillow has the highest average price/sqft being 907
- Following is townhouse and condo for approximately 755/sqft
- Multi family is just around 30/sqft less than townhouse and condo with the avg price/sqft being 715
- Manufactured is the cheapest home type in the bay area in term of price/sqft for 236/sqft''')

    with question3:
        st.image(
            "http://cdn.home-designing.com/wp-content/uploads/2014/07/small-3-bedroom-house-plan.jpeg",width = 600)
        #create a dataframe that have all data of houses that have 3 bedrooms and 2 bathrooms
        bed3bath2 = zillow_detail_df[(zillow_detail_df['bedrooms'] == 3) & (zillow_detail_df['bathrooms'] == 2)]
        st.write('The number of houses that have 3 bedrooms and 2 bathrooms are', bed3bath2.shape[0])
        #average the housing price for 3 bedrooms and 2 bathrooms
        price= bed3bath2.price.mean()
        def my_value(number):
            return ("{:,.0f}".format(number))
        st.markdown(f'The average price of a house that have 3 bedrooms and 2 bathrooms in the bay area is **${my_value(price)}**')


    with question4:
        st.write('---')
        q4_1,q4_2,q4_3 = st.columns((1,0.1,0.9))
        with q4_1:
            bed3bath2 = zillow_detail_df[(zillow_detail_df['bedrooms'] == 3) & (zillow_detail_df['bathrooms'] == 2)]
            pricehome = bed3bath2[['homeType','price']].groupby(['homeType']).mean().reset_index()
            pricehome = pricehome.sort_values(by='price', ascending=False)
            fig = ex.bar(data_frame=pricehome,x='homeType',y='price',hover_data=['homeType','price'],color ='price',labels={'homeType':'HomeType','price':'Average Price'},
                title = 'Average price for 3 bedrooms and 2 bathrooms houses by home type ')
            fig.update_layout(width = 650,height = 500,title_x = 0.5)
            st.plotly_chart(fig)

        with q4_3:
            st.markdown('''## Work:hammer::
- Create bed3bath2 dataframe contains all houses that have 3 bedrooms and 2 bathrooms
- Take homeType and price columns from bed3bath2, calculate average housing price and group them by home type.
- Plot a bar chart showing the Average Price for 3 bedrooms and 2 bathrooms by different home types

## Analysis:bulb::
- Multi family family is the most expensive home type for 3 beds and 2 baths house. It costs on average ~ 2.5 million dollars
- Following are single family and condo for approximately 1.3 and 1.1  million dollars, respectively for 3 beds and 2 baths
- Townhouse and Manufactured are the cheapest home type for 3 beds and 2 baths''')


    with question5:
        st.write('---')
        q5_1,q5_2,q5_3 = st.columns((1,0.1,0.8))
        with q5_1:
            pricecity = bed3bath2[['city','price']].groupby(['city']).mean().reset_index()
            pricecity = pricecity.sort_values(by='price', ascending=False)
            fig = ex.bar(data_frame=pricecity,x='city',y='price',hover_data=['city','price'],color ='price',labels={'city':'City','price':'Average Price(in $M)'},
            title = 'Average price for 3 bedrooms and 2 bathrooms in different cities in the Bay Area ')
            fig.update_layout(width = 775,height = 500,title_x = 0.5)
            st.plotly_chart(fig)

        with q5_3:
            st.markdown('''## Work:hammer::
- Create bed3bath2 dataframe contains all houses that have 3 bedrooms and 2 bathrooms
- Take city and price columns from houses, calculate average housing price and group them by city.
- Plot a bar chart showing the Average Price for 3 bedrooms and 2 bathrooms in different cities

## Analysis:bulb::
- Los Altos and Palo Alto are the most expensive cities in the bay area to buy 3 beds and 2 baths house. It costs on average ~ 3.5 million dollars
- For around 2 million dollars, you can buy 3 beds and 2 baths house in Sausalito, Santa Clara, Half Moon Bay, Orinda and San Mateo
- Hayward, Richmond, and San Pablo are the cheapest cities for buying 3 beds and 2 baths house with the housing price being less than 750,000''')


    with question6:
        st.write('---')
        q6_1,q6_2,q6_3 = st.columns((1,0.2,0.8))

        with q6_1:
            # Identifying the columns to be worked on from main dataframe
            mini_df = zillow_detail_df[['city','CalendarYear built']]

            # Arranging the data frame in the ascending oreder of 'CalendarYear built'
            mini_df = mini_df.sort_values(by='CalendarYear built', ascending=True)

            # Considering top 500 and bottom 500 properties to perform the analysis
            old_properties = mini_df.head(500)
            new_properties = mini_df.tail(500)

            # Identifying the top 10 cities with the oldest properties and newest properties
            old_properties = old_properties['city'].value_counts().rename_axis('City').reset_index(name='Number of properties').head(10)
            new_properties = new_properties['city'].value_counts().rename_axis('City').reset_index(name='Number of properties').head(10)

            # Plotting bar chart
            # Adjusting figure size and sharing y-axis.

            fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                            subplot_titles=("Cities with old properties","Cities with new properties"))

            fig.add_trace(go.Bar(x=old_properties['City'], y=old_properties['Number of properties'],
                            marker=dict(color=old_properties['Number of properties'],coloraxis="coloraxis")),
                      1, 1)

            fig.add_trace(go.Bar(x=new_properties['City'], y=new_properties['Number of properties'],
                            marker=dict(color=old_properties['Number of properties'],coloraxis="coloraxis")),
                      1, 2)

            fig.update_layout(showlegend=False,yaxis = dict(title="Number of Properties"),width = 800,height = 400)
            st.plotly_chart(fig)

        with q6_3:

            st.markdown('''#### Work:hammer::
- Removed unnecessary columns from the data frame
- Sorted the data frame based on 'CalendarYear built' in a descending order
- Considered top 500 rows and grouped them on city to analyze the number of old properties in a city
- Considered bottom 500 rows and grouped them on city to analyze the number of new properties in a city
- Plotted bar charts to show the number of old properties and new properties in a city (showing only top 10 cities)

#### Analysis:bulb::
- Berkeley has the most number of old properties listed on zillow to be sold
- Mountain view has the most number of new properties listed on zillow to be sold''')


    with question7:
        st.write('---')
        q7_1,q7_2,q7_3 = st.columns((1,0.2,0.8))

        with q7_1:
            @st.cache(allow_output_mutation=True)
            def f():
                return pd.read_csv("https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/final_df.csv")

            final_df = f()

            # Identifying the cities and their respective average priceChangePerYear
            by_city = final_df[['city','priceChangeRatePerYear']].groupby('city').mean()
            by_city = by_city.sort_values(by='priceChangeRatePerYear', ascending=False)
            by_city = by_city.reset_index()
            by_city = by_city.head(10)
            # Plotting bat chart for better understanding
            fig = ex.bar(data_frame=by_city,x='city',y='priceChangeRatePerYear',hover_data=['city','priceChangeRatePerYear'],
                    color ='priceChangeRatePerYear',labels={'city':'City','priceChangeRatePerYear':'Average PCR by Year'},
                    title = 'PriceChangeRate PerYear with respect to cities')
            fig.update_layout(width = 800,height = 400,title_x = 0.5)
            st.plotly_chart(fig)


            # Identifying the homeType and it's respective average priceChangeRate
            by_homeType = final_df[['homeType','priceChangeRatePerYear']].groupby('homeType').mean()
            by_homeType = by_homeType.sort_values(by='priceChangeRatePerYear', ascending=False)
            by_homeType = by_homeType.reset_index()

            # Plotting bat chart for better understanding
            fig1 = ex.bar(data_frame=by_homeType,x='homeType',y='priceChangeRatePerYear',hover_data=['homeType','priceChangeRatePerYear'],
                         color ='priceChangeRatePerYear',labels={'homeType':'HomeType','priceChangeRatePerYear':'Avg Price Change Rate'},
            title = 'PriceChangeRatePerYear with respect to Home Type')
            fig1.update_layout(width = 800,height = 400,title_x = 0.5,title_y = 0.9)
            st.plotly_chart(fig1)

        with q7_3:
            st.markdown('''#### Work:
- Removed all the rows and columns which are irrelavant for the analysis
- For each property, identified the price for which it was sold for the first time and the latest price of the property and calculated the price change rate
- Calculated the time gap(in years) between these two events
- Calculated the price change rate per year by dividing price change rate by time gap
- Grouped the resulted data frame based on city and hometype separately
- Plotted bar charts for 'price change rate per year' vs 'city' and 'home type' separately

#### Analysis:
- Danville has the highest appreciation rate of 35% per year
- Mill Valley has the lowest appreciation rate of 2% per year
- Prices of 'Multi family' type homes are increasing the most, which is 22% per year
- Prices of 'Single family' type homes are increasing 18% per year
- Prices of 'Town houses' type homes increasing 10% per year
- Prices of 'Condo' and 'Manufactured' type homes increasing almost 5% per year''')


    with question8:
        st.write("---")
        q8_1,q8_2 = st.columns((1,6))

        arg = "Crime"
        height = 500
        with q8_1:
            st.write("")
            st.write("")
            st.write("")
            arg = st.radio("",['Livability','Crime','Employment','Schools','Housing'])
            st.write("")
            agg = st.checkbox("NonAggregated")

        with q8_2:

            if agg:
                fig = ex.scatter(data_frame=zillow_detail_df,x=arg,y='price',size='price',size_max=15,
                color='price',height=height,width = 1000,title = f"Scatter plot representation of Price vs {arg}",
                hover_data=['bathrooms','bedrooms','price','city'])
                fig.update_layout(title_x = 0.5)
                st.plotly_chart(fig)
            else:
                if arg == 'Livability':
                    df = zillow_detail_df[['price',arg]].groupby(by=arg).mean().reset_index()
                else:
                    df = zillow_detail_df[['price',arg]].groupby(by=arg).mean().reset_index().sort_values(by=arg, key=lambda g: g + ',',ascending = False)
                fig = ex.scatter(data_frame=df,x=arg,y='price',size='price',size_max=20,
                color='price',height=height,width = 1000,title = f"Scatter Plot Representation Of Price Against {arg}",
                hover_data=['price',arg])
                fig.update_layout(title_x = 0.5)
                st.plotly_chart(fig)

        st.markdown('''### Work:hammer::
- Used pandas and matplotlib libraries
- After reading data from a CSV, mean of prices was taken with respect to cities and that column was merged to form a different data frame.
- Ratings data with respect to its average price change was maintained in lists
- This Categorical Data was then plotted using subplots
- These 3 subplots on the same x-axis help to get a better idea of price change with respect to crime ratings, school ratings and employment ratings of cities in the bay area.

### Analysis:bulb::
- Cities that have ratings in 'A' have a higher price range of houses.
- Areas with better Employment ratings matter the most and have highest ranges of prices.''')

        st.write("")



    with question9:

        st.markdown('''### Work 🔨:
- Grouped by cities and calculated average crime, school and employment ratings for each city
- Plotted bar charts for crime, school and employment ratings with respect to city

### Analysis💡:
- Los Gatos, Orinda, Saratoga,Los Altos, Lafayette, Danville, San Ramon, Tiburon are the **safest cities of Bay Area**.
- Los Altos, Saratoga, Tiburon, Palo Alto, Orinda, Alamo, Mill Valley, Danville, Menlo Park, Lafayette are the cities with highest employment ratings.
- Walnut Creek, Pleasanton, Mill Valley, Alamo, Los Gatos,Los Altos, Lafayette, Orinda, Palo Alto, Fremont, Dublin, Danville, San Carlos, Cupertino, San Ramon, Saratoga, Burlingame, Tiburon, Belmont are the cities with highest school ratings.

''')



    with question10:
        time4,time5,time6 = st.columns((0.35,0.05,1))


        with time4:
            c = st.radio("",['Counties','cities'])
            st.write("")
            st.markdown('''California is on of the hottest cities for real estate and growing strong day by day.
We have choose Bay Area as our reaserch area to analyse real estate market.
Below plot shows Average price for each of the counties and cities in california.
Grey Shaded Areas represent Economic recision and covid-19 pandemic respectively

**Note: San Francisco county average price is declining since lockdown**''')


        with time6:

            if c == 'Counties':

                county = st.multiselect("",counties.CountyName.unique(),counties.CountyName.unique()[:10])
                fig = ex.line(data_frame=counties[counties.CountyName.isin(county)],x='Year',y='MedianPrice',color='CountyName')
                fig.update_layout(yaxis = dict(title = "Average Price"))
                fig.add_vrect(x0="2008", x1="2009.75",
                annotation_text="Economic Recision", annotation_position="top left",
                fillcolor="black", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2020.1", x1="2021.25",
                annotation_text="Covid", annotation_position="top left",
                fillcolor="black", opacity=0.25, line_width=0)
                fig.update_layout(width = 950,height = 500)
                st.plotly_chart(fig)

            if c == 'cities':
                    #['Dublin','Danville','Alamo','San Francisco','Los Altos','Sunnyvale','Fremont']
                city = st.multiselect("",citiests.City.unique(),citiests.City.unique()[:10])
                fig = ex.line(data_frame = citiests[citiests.City.isin(city)],
                x='Year',y='MedianPrice',color = 'City')
                fig.update_layout(yaxis = dict(title = "Average Price"))
                fig.add_vrect(x0="2008", x1="2009.75",
                  annotation_text="Economic Recision", annotation_position="top left",
                  fillcolor="black", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2020.1", x1="2021.25",
                  annotation_text="Covid", annotation_position="top left",
                  fillcolor="black", opacity=0.25, line_width=0)
                fig.update_layout(width = 950,height = 500)
                st.plotly_chart(fig)




    with question12:
        st.write('---')
        pchange_df = df_joined
        pchange_df_before = df_joined

        # filtering rented house prices for data after March 19, 2020
        pchange_df = pchange_df[pchange_df['price_x']>10000.0]
        pchange_df = pchange_df[pchange_df['priceChangeRate']<100.0]

        # filtering rented house prices for data before March 19, 2020
        pchange_df_before = pchange_df_before[pchange_df_before['price_x']>10000.0]
        pchange_df_before = pchange_df_before[pchange_df_before['priceChangeRate']<100.0]

        # Data after March 19,2020 only
        pchange_df = pchange_df.loc[pchange_df['date'] >= '2020-03-19']
        pchange_df['date'] = pchange_df['date'].astype('|S')

        # Data between March 19,2018 and March 19,2020 only (2 years)
        pchange_df_before = pchange_df_before.loc[pchange_df_before['date'] < '2020-03-19']
        pchange_df_before = pchange_df_before.loc[pchange_df_before['date'] >= '2018-03-19']
        pchange_df_before['date'] = pchange_df_before['date'].astype('|S')

        # Calculating mean price change rate for each city
        price_change_rate_df = pchange_df.groupby(['city']).agg({'priceChangeRate':'mean'})
        price_change_rate_df = price_change_rate_df.reset_index()

        price_change_rate_df_before = pchange_df_before.groupby(['city']).agg({'priceChangeRate':'mean'})
        price_change_rate_df_before = price_change_rate_df_before.reset_index()

        #price_change_rate_df_before.sort_values(by = ['priceChangeRate'],inplace=True,ascending = False)

        fig = ex.bar(data_frame=price_change_rate_df_before,x='city',y='priceChangeRate',hover_data=['city','priceChangeRate'],
             color ='priceChangeRate',labels={'city':'City','priceChangeRate':'Average PriceChangeRate'},
            title = 'Average price change rate around bay areas before March 19, 2020 ')
        fig.update_layout(width = 1200,height = 600,title_x = 0.5)
        st.plotly_chart(fig)

        # price_change_rate_df.sort_values(by = ['priceChangeRate'],inplace=True,ascending = False)

        fig = ex.bar(data_frame=price_change_rate_df,x='city',y='priceChangeRate',hover_data=['city','priceChangeRate'],
                     color ='priceChangeRate',labels={'city':'City','priceChangeRate':'Average PriceChangeRate'},
                    title = 'Average price change rate around bay areas since March 19, 2020')
        fig.update_layout(width = 1200,height = 600,title_x = 0.5)
        st.plotly_chart(fig)

        price_change = pd.merge(price_change_rate_df,price_change_rate_df_before,how='outer',on = 'city')
        price_change.columns = ['City','PriceChangeRate_After','PriceChnageRate_Before']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=price_change.City,
            y=price_change.PriceChnageRate_Before,
            name='Before Lockdown',
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=price_change.City,
            y=price_change.PriceChangeRate_After,
            name='After Lockdown',
            marker_color='orange'
        ))

        # Here we modify the tickangle of the xaxis, resulting in rotated labels.
        fig.update_layout(barmode='group', xaxis_tickangle=90,width = 1250,height = 600,
        title_x = 0.5,title_text = "Average Price Change Rates Before[2018-2020] and After[2020-tilldate] the lockdown",
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),yaxis = dict(title="Average Prcie Change Rate"))
        st.plotly_chart(fig)

        st.markdown('''### Work:hammer::
- Joined PriceHistory and main_df dataset on Zpid.
- Created two dataframes, one with data of last 2 years (Mar 19,2018 till March 18,2020) and other with data during stay-at-home order (Mar 19,2020 onwards).
- Filtered rented house prices & outliers price change rate for both mentioned dataframes.
- changed the 'object' datatype to 'String' datatype of date column in both dataframes.
- In both dataframes, checked if any null values present in Price column, if so dropped the rows.
- For each city in bay area, calculated the average price change rate before and after March 19,2020.
- Plotted 3 graphs, first graph shows the average price change rate for each city before Lockdown, second graph shows the average price change rate for each city after Lockdown and third graph is the overlap of 2 graphs.

### Analysis:bulb::
- Comparing average price change rate for data before and over the period of stay-at-home (i.e, March 19, 2020) , it can be observed that almost all the cities changed their housing prices during lockdown, except Belmont, Berkeley, Burlingame, Cupertino, Los Altos, Los Gatos and Richmond where they decreased the same.''')

#=======================================================================================================================================================
#--------------------------------------------------------------Over view ------------------------------------------------------------------------
#=======================================================================================================================================================




if menubar == "Overview":
    st.subheader("Zillow Data")

    st.dataframe(zillow_detail_df,height = 350)

    st.write("")

    st.write("[Download data as CSV ](https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/main_df.csv)")

    st.title("Geo Distribution of Houses BayArea - California")

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
        st.subheader("Bubble Chart")

        fig = ex.scatter_mapbox(data[(data.notnull()['latitude'] & data.notnull()['price'])],
                                lat="latitude", lon="longitude", hover_name="city",
                                color = 'price', zoom=9, height=500,width = 650,size_max=12,size='price',hover_data = ['bedrooms','bathrooms','homeType'])
        fig.update_layout(mapbox_style="open-street-map")

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        st.plotly_chart(fig)


        st.markdown('''**This map show distribution of houses around california.
    The size of the bubble represents the house price.The larger the size of the bubble the higher the price of the house.The color of the bubble indicates
    in which range the house prices fall.Darker blue represents the house is of low price and yellow or orange represnet the houses of higher price.**''')

    with map2:
        st.subheader("Elevation  Chart")
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
        st.markdown('''**This Map show house Distribution in bay area.
    The height of each stick indicates how high the price of the house is.
    The color red indicates house is high price and green indicates that house is low price.
    The Map can be zoomed in and angle of view can be chnaged by mouse left click.**''')



#=========================================================================================================================================================
#----------------------------------------------------------House Price Predictions -----------------------------------------------------------------------
#=========================================================================================================================================================
if menubar == "House Price Predictions":

    st.title("House Price Prediction Using XGBoost")
    st.write("")

    pipeline1 = pickle.load(open('housemodel.pkl','rb'))

    zillow_detail_df = zillow_detail_df.set_index('zpid')
    regression_df = zillow_detail_df

    regression_df = regression_df[['city','price','bathrooms','bedrooms','livingArea',
                        'homeType','taxAssessedValue',
                        'Price Square FeetPrice/sqft','HasHeating','HasCooling','GarageSpaces','HasPool',
                         'FirePlaces','Crime', 'Employment', 'Schools']]

    regression_df = regression_df[regression_df.GarageSpaces<=6]
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
        # df.Housing = df['Housing'].map(ratings_dict)
        df.city = df['city'].map(city_dict)
        #
        return df

    test_data = pd.read_csv('https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/testing.csv')
    test_data.set_index('zpid',inplace = True)
    y_test = pd.DataFrame(zillow_detail_df.price,index=test_data.index)
    pred = pd.DataFrame(pipeline1.predict(test_data),index=test_data.index,columns = ['Predicted'])
    zest = pd.DataFrame(zillow_detail_df.zestimate,index=test_data.index)

    compare = pd.concat((y_test,np.round(pred),zest),axis = 1)


    test_res = pd.DataFrame({'Test r2_score':[round(r2_score(y_test,pred),2)],
                   'MSE Zillow Model':[np.sqrt(mean_squared_error(y_test,zest))],
                   'MSE XGB Model':[np.sqrt(mean_squared_error(y_test,pred))]
                   },index=[1])


    x4,x5,x6 = st.columns((1,0.1,1))
    with x4:

        st.markdown("**Test Data Metrics:**")
        st.markdown('''The metrics for the XGBoost model,r2 score close to 98 which indicates that model is able to capture
        Of the variance in y using the features. The mean square error for Zillow model is 156k  and mean square error of
        XGB model is 167k.
        ''')
        st.write('')
        st.write(test_res)

        st.markdown("""**Mean Square Error(MSE):**
- Mean Suqare Error is Sum of squares of residuals.
- Residuals is defined as difference between the listed price and predicted price.
- The Lower the mean square error means better model is.
""")



    with x6:
        st.write("")
        st.markdown('''As we can observe from the **Distribution of residuals** plot most of the residuals are around zero with few extremes having $1.5M and negative $1M''')
        fig = ex.histogram(compare['price']-compare['Predicted'])
        fig.update_layout(xaxis = dict(title = 'Error'),title = 'Distribution of residuals',title_x = 0.5,title_y = 0.9)
        st.plotly_chart(fig)

    diff_plot = pd.concat((pd.DataFrame(compare['price']-compare['Predicted'],columns = ['PriceDifference']),zillow_detail_df[['latitude','longitude']].reindex(index=  test_data.index)),axis =1)
    diff_plot = pd.concat((pd.DataFrame(abs(compare['price']-compare['Predicted']),columns = ['AbsPriceDifference']),
                                   diff_plot),axis = 1)

    x1,x2,x3 = st.columns((1,0.1,0.8))
    with x1:
        fig = ex.scatter_mapbox(data_frame=diff_plot,
                            lat="latitude", lon="longitude", hover_name="PriceDifference",
                            color = 'PriceDifference', zoom=9, height=600,width = 800,size_max=25,size = 'AbsPriceDifference')
        fig.update_layout(mapbox_style="open-street-map")

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)
    with x3:
        st.subheader("Difference Between the predicted and listed price")
        st.markdown('''The size of the of the circle represents
The magnitude of the error we made
While predicting the house price.

If the circle color is Dark Blue that means
Model has over estimated the price.

If the circle color is Closer to orange or
Yellow it indicates that we have under
Estimated the price of the house.
''')

    st.write("---")

    st.subheader("Predict House prices using the test data")

    i = np.random.randint(10,len(test_data)-10)
    one = (zillow_detail_df.loc[test_data.index,:]).iloc[i:i+1,:]
    st.write(one)

    but1 = st.button("Predict house price for test data with user model",key=1000)

    if but1:
        st.write("")
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

    st.header("Predict House Price with your own customized Data")

    userinputs = st.expander("Predict house prices with your own customized data")

    with userinputs:
        st.write("---")


        i1,i2,i3 = st.columns((1,1,1))
        i4,i5,i6 = st.columns((1,1,1))
        i7,i8,i9 = st.columns((1,1,1))
        i10,i11,i12 = st.columns((1,1,1))
        i13,i14,i15 = st.columns((1,1,1))

        input_list = [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]

        d = {}
        for j,i in enumerate(reg_df.columns):
            if reg_df[i].dtype == 'object':
                with input_list[j]:
                    v = st.selectbox(i.upper(),sorted(list(reg_df[i].unique())),key = i)

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
            st.subheader(f'The predicted house price with specifications cost approximateley ${np.round(pipeline1.predict(format(input)))[0]}')





if menubar == "About Us":
    i1,i2,i3 = st.columns((1,1,1))
    with i1:
        st.write("")
        st.markdown('''# Group Members :

- Deepthi Mounica Mandalaparty
- Kavya	Pothula
- Prathamesh Bhople
- Rosie	Nguyen
- Sayali Mahamulkar
- Vishnu Nelapati''')

    with i2:
        st.image('https://www.pngall.com/wp-content/uploads/9/Happy-Minions-PNG-Images.png',width = 300)
