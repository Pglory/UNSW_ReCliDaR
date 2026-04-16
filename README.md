ReCliDaR: Representative Climate Days Recognizer
by University of New South Wales Sydney

ReCliDaR is a research tool designed to identify representative climate days from EnergyPlus Weather (.epw) files using multi-objective machine learning clustering. It evaluates climate patterns using k-Means, Gaussian Mixture Models (GMM), and Agglomerative Clustering (HAC) to provide high-fidelity representative data for building performance simulations.

🚀 Access and Usage
1. Web Interface (Fastest)
The web application provides an instant, browser-based analysis tool for researchers who want to process individual files without any installation.

Access: [https://unsw-reclidar.streamlit.app/]

Usage:

Upload your .epw file.

Click Run ML Analysis.

Download the Representative Days Report and Monthly Distribution CSV files.

Best for: Quick analysis, non-programmers, and cross-platform (Mac/Linux/Windows) access.

2. Windows Executable (Offline)
For users who prefer a local, standalone application that works without a Python environment or internet connection.

Access: Go to the Releases section of this repository.

Download: ReCliDaR_tool.exe (approx. 126 MB).

Usage: Double-click the file to launch the Graphical User Interface (GUI). No installation is required.

Best for: Offline research and users who prefer a traditional desktop experience.

3. Python Research Engine (Advanced/Batch)
The reclidar_engine.py script is a clean, modular class designed for developers and researchers who need to process large-scale datasets (hundreds of EPW files) within their own Python pipelines.

Access: Download reclidar_engine.py from the main repository folder.

Dependencies: Ensure you have installed the required libraries:
pip install streamlit ladybug-core pandas numpy scikit-learn Pillow

Usage Example:

Python
from reclidar_engine import ReCliDaR

# Initialize and process
orig, scaled = ReCliDaR.process_epw("sydney_climate.epw")

# Run specific algorithm (kMeans, GMM, or HAC)
labels = ReCliDaR.run_analysis(scaled, method='kMeans')

# Get the two primary research outputs
reps_df = ReCliDaR.get_representative_days(orig, scaled, labels, 'kMeans')
dist_df = ReCliDaR.get_monthly_distribution(orig, labels, 'kMeans')
Best for: Large-scale batch processing, integration into Grasshopper/Blender workflows, and customized academic research.

📊 Outputs
Regardless of the method used, the tool generates two primary datasets:

Representative Days: A summary of the typical days identified for each cluster, including their month, day, and the "weight" (number of days) they represent in the full year.

Monthly Distribution: A matrix showing the frequency of each cluster type across the 12 months of the year.

⚖️ License & Citation
Distributed under the MIT License. If you use this tool in your research, please cite:
Naga, Faculty of Architecture and Town Planning, UNSW Sydney.
