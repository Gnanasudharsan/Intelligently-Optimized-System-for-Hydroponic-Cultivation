# Intelligently Optimized System for Hydroponic Cultivation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Arduino](https://img.shields.io/badge/Arduino-IDE-blue.svg)](https://www.arduino.cc/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red.svg)](https://www.raspberrypi.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/)

An automated hydroponic system that uses IoT sensors, machine learning, and computer vision to optimize plant growth conditions. This system monitors and controls pH levels, nutrient concentration (TDS), temperature, humidity, and provides disease detection capabilities.

![System Overview](docs/images/system_overview.jpg)

## 🌱 Features

- **Automated Monitoring**: Real-time monitoring of pH, TDS, temperature, and humidity
- **IoT Integration**: Data logging to ThingSpeak cloud platform
- **Disease Detection**: Computer vision-based plant disease identification using OpenCV
- **Preset Management**: Customizable growing presets for different plant types
- **Remote Access**: Web dashboard for monitoring and control
- **Nutrient Film Technique**: Efficient hydroponic growing method implementation
- **Scalable Design**: Modular system that can be expanded

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Sensors   │───▶│ Arduino Uno  │───▶│Raspberry Pi │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│  Actuators  │◀───│   Presets    │◀──────────┤
└─────────────┘    └──────────────┘           │
                                              ▼ 
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Dashboard │◀───│  ThingSpeak  │◀───│   WebCam    │
└─────────────┘    └──────────────┘    └─────────────┘
```

## 🔧 Hardware Components

### Sensors
- **pH Sensor**: Gravity Analog pH sensor for solution acidity monitoring
- **TDS Sensor**: Total Dissolved Solids sensor for nutrient monitoring
- **DHT11**: Temperature and humidity sensor
- **MAX6675**: Thermocouple module for solution temperature
- **USB Webcam**: For plant monitoring and disease detection

### Controllers
- **Arduino Uno**: Sensor data collection and actuator control
- **Raspberry Pi 4 Model B**: Main processing unit and IoT gateway

### Actuators
- Water pumps for nutrient circulation
- LED grow lights (Red, Blue, White spectrum)
- pH adjustment pumps
- Nutrient dosing pumps

## 🚀 Quick Start

### Prerequisites
- Arduino IDE
- Python 3.7+
- Raspberry Pi OS
- ThingSpeak account

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Gnanasudharsan/hydroponic-cultivation-system.git
cd hydroponic-cultivation-system
```

2. **Arduino Setup**
```bash
# Upload the Arduino code
# Open arduino/hydroponic_sensors/hydroponic_sensors.ino in Arduino IDE
# Install required libraries (see arduino/libraries.txt)
# Upload to Arduino Uno
```

3. **Raspberry Pi Setup**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure ThingSpeak credentials
cp config/config_template.py config/config.py
# Edit config.py with your ThingSpeak API keys

# Run the main application
python raspberry_pi/main.py
```

## 📁 Project Structure

```
hydroponic-cultivation-system/
│
├── arduino/
│   ├── hydroponic_sensors/
│   │   └── hydroponic_sensors.ino
│   ├── libraries.txt
│   └── README.md
│
├── raspberry_pi/
│   ├── main.py
│   ├── sensor_reader.py
│   ├── disease_detection.py
│   ├── thingspeak_client.py
│   ├── preset_manager.py
│   └── actuator_controller.py
│
├── web_dashboard/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── README.md
│
├── presets/
│   ├── lettuce.json
│   ├── tomato.json
│   ├── spinach.json
│   └── amaranthus.json
│
├── docs/
│   ├── images/
│   ├── fritzing/
│   ├── installation_guide.md
│   ├── api_documentation.md
│   └── troubleshooting.md
│
├── config/
│   ├── config_template.py
│   └── thingspeak_config.json
│
├── tests/
│   ├── test_sensors.py
│   ├── test_disease_detection.py
│   └── test_actuators.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

## 🎛️ Usage

### 1. System Monitoring
Access the web dashboard at `http://raspberry-pi-ip:5000` to view:
- Real-time sensor data
- Historical trends
- Plant growth analytics
- Disease detection alerts

### 2. Plant Presets
Select from pre-configured plant presets or create custom ones:
```python
# Example preset for lettuce
{
    "plant_name": "lettuce",
    "ph_range": [5.5, 6.5],
    "ec_range": [0.8, 1.2],
    "temperature_range": [18, 24],
    "humidity_range": [50, 70],
    "light_schedule": {
        "on_hours": 14,
        "off_hours": 10
    }
}
```

### 3. Disease Detection
The system automatically analyzes plant images and alerts for:
- Leaf discoloration
- Disease symptoms
- Growth abnormalities

Accuracy: **85-87%** based on testing with Amaranthus caudatus

## 📈 Results & Performance

- **Water Efficiency**: 90% reduction compared to traditional farming
- **Growth Rate**: 30-50% faster than soil-based cultivation
- **Disease Detection Accuracy**: 85-87%
- **Automation Level**: Fully automated with minimal human intervention
- **Scalability**: Modular design supports multiple growing units

## 🔬 Research Paper

This project is based on our research paper: "Intelligently Optimized System for Hydroponic Cultivation" presented at the 2022 International Conference on Communication, Computing and Internet of Things (IC3IoT).

**Authors**: Yashwanth D, Pooja R, Keerthika M, G Prasanth, Gnanasudharsan A, Sandeep V, Shabana Parveen M

## 🛠️ API Documentation

### ThingSpeak Integration
```python
# Send sensor data
POST https://api.thingspeak.com/update
{
    "api_key": "YOUR_API_KEY",
    "field1": ph_value,
    "field2": tds_value,
    "field3": temperature,
    "field4": humidity
}
```

### Disease Detection API
```python
# Analyze plant image
POST /api/analyze_plant
{
    "image": "base64_encoded_image"
}

Response:
{
    "disease_detected": boolean,
    "confidence": float,
    "disease_type": string,
    "recommendations": [array]
}
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Individual test categories:
```bash
# Test sensors
python -m pytest tests/test_sensors.py

# Test disease detection
python -m pytest tests/test_disease_detection.py

# Test actuators
python -m pytest tests/test_actuators.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sri Sairam Engineering College, Chennai
- IEEE for publishing our research
- ThingSpeak platform for IoT integration
- OpenCV community for computer vision tools

## 📞 Support

For support and questions:
- Create an [issue](https://github.com/yourusername/hydroponic-cultivation-system/issues)
- Email: yashwanth.devadoss@gmail.com

## 🔮 Future Scope

- [ ] Mobile application development
- [ ] Integration with more cloud platforms
- [ ] Advanced ML models for yield prediction
- [ ] Support for mushroom and moss cultivation
- [ ] Multi-language support
- [ ] Commercial restaurant integration

---

**⭐ Star this repository if you found it helpful!**
