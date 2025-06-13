# SyntheticDataUI 🚀✨

SyntheticDataUI is a Django-based web application for generating synthetic data. It was developed as part of a Field Project for BSC Data Science & Society at the University of Groningen. 

The goal of the application is to provide a user-friendly interface for uploading CSV files, generating synthetic records using agent-based modelling, and downloading the results.

The project was developed for the Municipality of Leeuwarden. It is intended to be used by data scientists and analysts who need to create synthetic datasets for testing, training, or privacy-preserving purposes.

![Gemeente Leeuwarden Logo](https://www.wierenga-degraaf.nl/beheer/wp-content/uploads/2020/03/leeuwarden-logo.png)

## Features 🎉
- 📤 Upload a CSV file and edit the data directly in the browser.
- 🤖 Generate synthetic records using the algorithms defined in `synth_ui/frontend/synth.py`.
- 📥 Download the generated synthetic data or evaluation metrics.

## Warning ⚠️
This project was specifically developed for the Municipality of Leeuwarden and may not be suitable for other use cases without modifications. The synthetic data generation algorithms are tailored to the specific requirements and data structures of the Leeuwarden dataset.

For the future, we hope someone will take this project and generalize it to support a wider range of datasets and use cases.

## Setup 🛠️🔧

### Create the Conda environment 🐍
1. Install [Conda](https://docs.conda.io/) if you don't already have it.
2. Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

This will create an environment named **SYNTHETIC** with all required dependencies.
Activate it with:

```bash
conda activate SYNTHETIC
```

### Database migrations 🗄️
Run the initial migrations to set up the SQLite database:

```bash
python synth_ui/manage.py makemigrations
```

Then apply the migrations:

```bash
python synth_ui/manage.py migrate
```

### Launch the development server 🚀
Start the Django development server:

```bash
python synth_ui/manage.py runserver
```

Then open your browser to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to upload a CSV and try generating synthetic records.

## Usage 🔧
1. 🌐 Navigate to the web application in your browser.
2. 📂 Upload a CSV file containing the data you want to synthesize.
3. ✏️ Edit the data directly in the browser if needed.
4. 🧪 Click the “Generate Synthetic Data” button to create synthetic records.
5. 📊 Download the generated synthetic data or evaluation metrics as needed.

## Contributing 🤝✨
Contributions are welcome! If you find any issues or have suggestions for improvements, please:

- 🐛 Open an issue
- 💡 Submit a pull request
- ⭐️ Give the project a star if you like it!

## Future Work 🔮🚧
Future enhancements may include:
- 📊 Support for additional data formats (e.g., Excel, JSON).
- 🤖 More advanced synthetic data generation algorithms.
- 🔗 A more generalized approach to handle different types of data and relationships.
- 🛠️ A user-interface that allows for more complex configurations and settings.

## Acknowledgements 🙏
This project was developed as part of a Field Project for BSC Data Science & Society at the University of Groningen. Special thanks to the Municipality of Leeuwarden for providing the context and requirements for this application.

## License ⚖️
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

