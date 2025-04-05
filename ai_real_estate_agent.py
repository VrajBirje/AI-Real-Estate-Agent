from typing import Dict, List
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
import streamlit as st
from openai import OpenAI


class CustomOpenAIWrapper:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return type("Obj", (object,), {"content": response.choices[0].message.content})


class PropertyData(BaseModel):
    building_name: str = Field(description="Name of the building/property", alias="Building_name")
    property_type: str = Field(description="Type of property (commercial, residential, etc)", alias="Property_type")
    location_address: str = Field(description="Complete address of the property")
    price: str = Field(description="Price of the property", alias="Price")
    description: str = Field(description="Detailed description of the property", alias="Description")


class PropertiesResponse(BaseModel):
    properties: List[PropertyData] = Field(description="List of property details")


class LocationData(BaseModel):
    location: str
    price_per_sqft: float
    percent_increase: float
    rental_yield: float


class LocationsResponse(BaseModel):
    locations: List[LocationData] = Field(description="List of location data points")


class PropertyFindingAgent:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str, model_id: str = "gpt-4"):
        self.agent = CustomOpenAIWrapper(api_key=openai_api_key, model=model_id)
        try:
            self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        except Exception as e:
            st.error(f"Failed to initialize Firecrawl: {str(e)}")
            raise

    def find_properties(self, city: str, max_price: float, property_category: str = "Residential", property_type: str = "Flat") -> str:
        try:
            formatted_location = city.lower()
            urls = [
                f"https://www.squareyards.com/sale/property-for-sale-in-{formatted_location}/*",
                f"https://www.99acres.com/property-in-{formatted_location}-ffid/*",
                f"https://housing.com/in/buy/{formatted_location}/{formatted_location}"
            ]

            property_type_prompt = "Flats" if property_type == "Flat" else "Individual Houses"

            raw_response = self.firecrawl.extract(
                urls=urls,
                params={
                    'prompt': f"""Extract ONLY 10 OR LESS different {property_category} {property_type_prompt} from {city} that cost less than {max_price} crores.
                    
                    Requirements:
                    - Property Category: {property_category} properties only
                    - Property Type: {property_type_prompt} only
                    - Location: {city}
                    - Maximum Price: {max_price} crores
                    - Include complete property details with exact location
                    - IMPORTANT: Return data for at least 3 different properties. MAXIMUM 10.
                    - Format as a list of properties with their respective details
                    """,
                    'schema': PropertiesResponse.model_json_schema()
                }
            )

            if isinstance(raw_response, dict) and raw_response.get('success'):
                properties = raw_response['data'].get('properties', [])
            else:
                properties = []

            analysis = self.agent.run(
                f"""As a real estate expert, analyze these properties and market trends:

                Properties Found:
                {properties}

                **IMPORTANT INSTRUCTIONS:**
                1. ONLY analyze properties from the above data that match:
                   - Property Category: {property_category}
                   - Property Type: {property_type}
                   - Maximum Price: {max_price} crores
                2. Select 5-6 properties with prices closest to {max_price} crores

                Provide analysis in this format:
                
                🏠 SELECTED PROPERTIES
                • List 5-6 best matching properties
                • For each include:
                  - Name and Location
                  - Price analysis
                  - Key Features
                  - Pros and Cons

                💰 BEST VALUE ANALYSIS
                • Compare properties on:
                  - Price per sq ft
                  - Location
                  - Amenities

                📍 LOCATION INSIGHTS
                • Area advantages

                💡 RECOMMENDATIONS
                • Top 3 properties with reasoning
                • Investment potential
                • Purchase considerations

                🤝 NEGOTIATION TIPS
                • Property-specific strategies
                """
            )

            return analysis.content

        except Exception as e:
            st.error(f"Error in find_properties: {str(e)}")
            return f"Error: {str(e)}"

    def get_location_trends(self, city: str) -> str:
        try:
            raw_response = self.firecrawl.extract([
                f"https://www.99acres.com/property-rates-and-price-trends-in-{city.lower()}-prffid/*"
            ], {
                'prompt': """Extract price trends data for ALL major localities in the city. 
                IMPORTANT: 
                - Return data for at least 5-10 different localities
                - Include both premium and affordable areas
                - Do not skip any locality mentioned in the source
                - Format as a list of locations with their respective data
                """,
                'schema': LocationsResponse.model_json_schema(),
            })

            if isinstance(raw_response, dict) and raw_response.get('success'):
                locations = raw_response['data'].get('locations', [])

                analysis = self.agent.run(
                    f"""Analyze these location price trends for {city}:

                    {locations}

                    Provide:
                    1. Summary of price trends for each location
                    2. Top 3 locations for:
                       - Price appreciation
                       - Rental yields
                       - Value for money
                    3. Investment recommendations:
                       - Long-term investments
                       - Rental income
                       - Emerging areas
                    4. Specific investor advice

                    Format:
                    
                    📊 LOCATION TRENDS SUMMARY
                    • [Bullet points]

                    🏆 TOP PERFORMING AREAS
                    • [Bullet points]

                    💡 INVESTMENT INSIGHTS
                    • [Bullet points]

                    🎯 RECOMMENDATIONS
                    • [Bullet points]
                    """
                )

                return analysis.content

            return "No price trends data available"

        except Exception as e:
            st.error(f"Error in get_location_trends: {str(e)}")
            return f"Error: {str(e)}"


def main():
    st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="wide")

    with st.sidebar:
        st.title("🔑 API Configuration")
        
        st.subheader("🔐 API Keys")
        firecrawl_key = st.text_input("Firecrawl API Key", type="password", help="Get your key from https://firecrawl.dev")
        openai_key = st.text_input("OpenAI API Key", type="password", help="Get your key from https://platform.openai.com")
        
        st.subheader("🤖 Model Selection")
        model_id = st.selectbox(
            "Choose OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=1,
            help="Select the AI model to use"
        )

    st.title("🏠 AI Real Estate Agent")
    st.info("Enter your search criteria to get property recommendations and location insights.")

    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("City", placeholder="e.g., Bangalore", help="Enter the city to search")
        property_category = st.selectbox("Property Category", options=["Residential", "Commercial"])

    with col2:
        max_price = st.number_input("Maximum Price (in Crores)", min_value=0.1, value=5.0, step=0.1)
        property_type = st.selectbox("Property Type", options=["Flat", "Individual House"])

    if st.button("🔍 Start Search", use_container_width=True):
        if not firecrawl_key or not openai_key:
            st.error("Please enter both API keys in the sidebar")
            return
            
        if not city:
            st.error("Please enter a city name")
            return

        try:
            agent = PropertyFindingAgent(
                firecrawl_api_key=firecrawl_key,
                openai_api_key=openai_key,
                model_id=model_id
            )

            with st.spinner("🔍 Searching for properties..."):
                property_results = agent.find_properties(
                    city=city,
                    max_price=max_price,
                    property_category=property_category,
                    property_type=property_type
                )
                
                st.success("✅ Property search completed!")
                st.subheader("🏘️ Property Recommendations")
                st.markdown(property_results)

                st.divider()

                with st.spinner("📊 Analyzing location trends..."):
                    location_trends = agent.get_location_trends(city)
                    st.success("✅ Location analysis completed!")
                    
                    with st.expander("📈 Location Trends Analysis"):
                        st.markdown(location_trends)

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()