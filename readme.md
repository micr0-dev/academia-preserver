# Academia.edu Paper Preserver

A robust Python tool for preserving academic papers from Academia.edu through automated downloading. This tool allows researchers and archivists to bulk download papers based on keywords, with features for resuming interrupted downloads and managing large collections.

## Purpose

This tool was created to help:
- Preserve academic research that might become unavailable
- Create local archives of research papers for offline access
- Batch download papers related to specific research topics
- Ensure research accessibility and preservation

## Features

- **Multiple Keyword Search**: Download papers from multiple research areas simultaneously
- **Resume Capability**: Safely interrupt and resume downloads
- **Progress Tracking**: Visual progress bars for both overall and individual download progress
- **Smart Naming**: Organized filename structure including author, title, and keywords
- **User-Friendly Interface**: Clear prompts and progress indicators
- **State Preservation**: Maintains download state for recovery from interruptions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/micr0-dev/academia-preserver.git
cd academia-preserver
```

2. Install required packages:
```bash
pip install requests tqdm rich aiohttp
```

## Usage

1. Run the script:
```bash
python main.py
```

2. Follow the interactive prompts:
   - Enter search keywords (comma-separated)
   - Specify output directory
   - Choose maximum number of papers to download

### Example Input:

```
Enter search keywords: transgender, queer theory, gender studies
Enter output directory: my_papers
Maximum number of papers to download: all
```

### Resuming Downloads

If a download is interrupted, you can resume it:
1. Run the script again
2. Choose 'yes' when asked about resuming
3. The download will continue from where it left off

## File Organization

Downloaded papers are saved with the following naming convention:
```
AuthorName-PaperTitle-Keyword-UniqueID.extension
```

Example:
```
John_Smith-Gender_Theory_Analysis-transgender-a1b2c3d4.pdf
```

## Important Notes

- Be mindful of Academia.edu's terms of service
- Consider rate limiting and server load
- Large downloads might take significant time
- Ensure adequate disk space for downloads
- Check your internet connection stability for large downloads

## Error Handling

The script handles various error conditions:
- Network interruptions
- Server timeouts
- Invalid responses
- File system issues

If an error occurs, the script will:
1. Save the current state
2. Display error information
3. Allow resuming from the last successful download

## Contributing

Contributions are welcome! Please feel free to submit pull requests with improvements or bug fixes.

## License

This project is licensed under the [OVERWORKED LICENSE (OWL) v1.0.](https://owl-license.org/) See the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for research and archival purposes. Users are responsible for ensuring compliance with Academia.edu's terms of service and respecting copyright restrictions.