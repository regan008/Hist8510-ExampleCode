# Week 10: Semantic Similarity Analysis

## Files:
- `semantic_similarity_analysis.py` - Semantic analysis for LGBTQ locations
- `analyze_inaugural_addresses.py` - Semantic analysis for inaugural addresses
- `scrape_inaugural_addresses.py` - Web scraper for inaugural addresses
- `data_philly.csv` - Location data from LGBTQ guidebooks
- `requirements.txt` - Python dependencies
- `setup.sh` - Setup script
- `reset.sh` - Clean analysis results
- `semantic_similarity_results.csv` - Analysis output (generated)

## Setup:

1. Run the setup script:
   ```bash
   ./setup.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run LGBTQ location analysis:
   ```bash
   python semantic_similarity_analysis.py
   ```

4. Run inaugural address analysis:
   ```bash
   cd inagural_addresses
   python analyze_inaugural_addresses.py
   ```

5. Deactivate when done:
   ```bash
   deactivate
   ```

## Reset analysis results:
```bash
cd inagural_addresses
./reset.sh
```