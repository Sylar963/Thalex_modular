# Thalex Avellaneda Market Maker

A sophisticated market making bot for Thalex exchange implementing the Avellaneda-Stoikov model.

## Features

- Avellaneda-Stoikov market making strategy
- Dynamic spread calculation based on volatility and inventory
- Advanced risk management with multiple take-profit levels
- Technical analysis integration
- Comprehensive position management
- Robust error handling and recovery

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Thalex_SimpleQuouter.git
cd Thalex_SimpleQuouter

# Install dependencies
pip install -e .
```

## Configuration

1. Copy the template configuration file:
```bash
cp thalex_py/Thalex_modular/config/keys_template.py thalex_py/Thalex_modular/config/keys.py
```

2. Edit `keys.py` with your Thalex API credentials.

## Usage

```bash
# Run the market maker
python -m thalex_py.Thalex_modular.avellaneda_quoter
```

## Project Structure

```
thalex_py/
└── Thalex_modular/
    ├── avellaneda_quoter.py      # Main market maker
    ├── components/               # Core components
    ├── config/                   # Configuration
    └── models/                   # Data models
```

## Documentation

See the `docs/` directory for detailed documentation.

## Testing

```bash
# Run tests
python -m pytest tests/
```

## License

MIT License 