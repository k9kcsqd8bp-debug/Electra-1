def estimate_cost(energy_kwh, tariff_per_kwh):
    """Illustrative estimate only; excludes taxes, fixed charges, slabs and other utility adjustments."""
    return float(energy_kwh) * float(tariff_per_kwh)
