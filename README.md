# 🍏🍾 Sidrobus 🚌🔋⚡️🔌 - Electric and Fuel Bus Simulator

![](logo.png)

> "On February 28, 1956, the first two buses arrived in Oviedo to begin replacing the tram. These were two Pegasos that operated on the Colloto - Cruce de las Caldas line, and the local humor dubbed them **sidrobuses** due to their popular use for going to drink sidra (cider) in Colloto." - [Transportes Unidos de Asturias, Historia](https://www.tua.es/es/tua/historia/)

## Author

**Daniel Castaño Rodríguez**

[GitHub](https://github.com/Daniel-C-R/)

[LinkedIn](https://www.linkedin.com/in/danielcr1/)

[Email](mailto:daniel.cr.0001@gmail.com)

[ORCID](https://orcid.org/0009-0009-6907-7798)

## Overview

This project is an electric and combustion bus simulator designed to help municipalities and urban transport operators assess the feasibility of electrifying their fleets. The simulator allows you to configure the parameters of a bus and calculate energy consumption and pollutant emissions (for combustion buses), or use one of the pre-implemented models. You can also create routes to physically simulate the bus journey and obtain results such as energy consumption and emissions. This can be done either through an API that can be imported from Python or via a web application developed with Streamlit.

The project was developed as a Bachelor's Thesis for the Data Science and Engineering degree at the University of Oviedo.

You can use the application directly at the following link: [https://sidrobus.streamlit.app/](https://sidrobus.streamlit.app/).

## Running the Application

The application is managed with Poetry, so you need to have it installed to run the project. Once installed, follow these steps:

1. **Install dependencies**: Run the following command in your terminal to install all project dependencies:

    ```bash
    poetry install
    ```

2. **Run the application**: Once the dependencies are installed, launch the Streamlit app with:

    ```bash
    poetry run streamlit run web_app/Sidrobus.py
    ```

## Disclaimer

While every effort has been made to ensure the accuracy of the calculations and information provided by this application, the results are offered without any warranty, express or implied. The author accepts no responsibility for any errors, omissions, or consequences arising from the use of the application. Users agree to use the information at their own risk.
