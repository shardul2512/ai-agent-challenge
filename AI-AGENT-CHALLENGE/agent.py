#!/usr/bin/env python3
"""Karbon AI Agent-as-Coder
Uses LLM to generate and self-fix bank statement parsers.
Supports PDF, Excel (XLSX/XLS), and CSV inputs → Always outputs CSV

Usage:
  python agent.py --bank icici --input "icici sample.pdf" --expected result.csv
  python agent.py --bank sbi --input "sbi_statement.xlsx" --expected result.csv
  python agent.py --bank hdfc --input "hdfc_data.csv" --expected result.csv
  
Environment variables:
  GEMINI_API_KEY or GROQ_API_KEY
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Literal
import pandas as pd

# Try to import LLM libraries
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class LLMClient:
    """Unified LLM client supporting Gemini and Groq."""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self._setup()
    
    def _setup(self):
        """Initialize available LLM provider."""
        # Try Gemini first
        gemini_key = os.getenv('GEMINI_API_KEY')
        if HAS_GEMINI and gemini_key:
            genai.configure(api_key=gemini_key)
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            self.provider = 'gemini'
            print('[Agent] Using Gemini API')
            return
        
        # Try Groq
        groq_key = os.getenv('GROQ_API_KEY')
        if HAS_GROQ and groq_key:
            self.client = Groq(api_key=groq_key)
            self.provider = 'groq'
            print('[Agent] Using Groq API')
            return
        
        raise RuntimeError(
            "No LLM provider available. Please:\n"
            "1. Install: pip install google-generativeai OR pip install groq\n"
            "2. Set GEMINI_API_KEY or GROQ_API_KEY environment variable"
        )
    
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        if self.provider == 'gemini':
            response = self.client.generate_content(prompt)
            return response.text
        elif self.provider == 'groq':
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            return chat.choices[0].message.content
        raise RuntimeError("No provider initialized")


class FileAnalyzer:
    """Analyzes input files and extracts sample content."""
    
    @staticmethod
    def detect_format(file_path: Path) -> Literal['pdf', 'excel', 'csv']:
        """Detect input file format."""
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return 'pdf'
        elif suffix in ['.xlsx', '.xls', '.xlsm']:
            return 'excel'
        elif suffix == '.csv':
            return 'csv'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def extract_sample(file_path: Path, file_format: str) -> dict:
        """Extract sample content based on file format."""
        if file_format == 'pdf':
            return FileAnalyzer._extract_pdf_sample(file_path)
        elif file_format == 'excel':
            return FileAnalyzer._extract_excel_sample(file_path)
        elif file_format == 'csv':
            return FileAnalyzer._extract_csv_sample(file_path)
    
    @staticmethod
    def _extract_pdf_sample(file_path: Path) -> dict:
        """Extract sample text from PDF."""
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                sample_text = ""
                total_pages = len(pdf.pages)
                
                # Extract from first 2 pages
                for page in pdf.pages[:2]:
                    sample_text += page.extract_text() or ""
                
                return {
                    'format': 'pdf',
                    'pages': total_pages,
                    'sample': sample_text[:2000],
                    'extraction_method': 'text_extraction'
                }
        except Exception as e:
            return {
                'format': 'pdf',
                'error': str(e),
                'sample': f"[Error reading PDF: {e}]"
            }
    
    @staticmethod
    def _extract_excel_sample(file_path: Path) -> dict:
        """Extract sample data from Excel."""
        try:
            import openpyxl
            # Read first sheet
            df = pd.read_excel(file_path, nrows=10)
            
            # Get all sheet names
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            
            return {
                'format': 'excel',
                'sheets': sheet_names,
                'sample_shape': df.shape,
                'sample_columns': list(df.columns),
                'sample_data': df.head(5).to_dict('records'),
                'sample_text': df.head(5).to_string(),
                'extraction_method': 'pandas_excel'
            }
        except Exception as e:
            return {
                'format': 'excel',
                'error': str(e),
                'sample': f"[Error reading Excel: {e}]"
            }
    
    @staticmethod
    def _extract_csv_sample(file_path: Path) -> dict:
        """Extract sample data from CSV."""
        try:
            # Try reading with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, nrows=10, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            return {
                'format': 'csv',
                'sample_shape': df.shape,
                'sample_columns': list(df.columns),
                'sample_data': df.head(5).to_dict('records'),
                'sample_text': df.head(5).to_string(),
                'extraction_method': 'pandas_csv'
            }
        except Exception as e:
            return {
                'format': 'csv',
                'error': str(e),
                'sample': f"[Error reading CSV: {e}]"
            }


class AgentAsCoder:
    """AI Agent that writes and fixes bank statement parsers."""
    
    def __init__(self, bank: str, input_path: Path, expected_csv: Path):
        self.bank = bank
        self.input_path = input_path
        self.expected_csv = expected_csv
        self.parser_path = Path(f'custom_parsers/{bank}_parser.py')
        self.llm = LLMClient()
        self.history = []
        
        # Detect input format
        self.input_format = FileAnalyzer.detect_format(input_path)
        print(f'[Agent] Detected input format: {self.input_format.upper()}')
        
    def run(self) -> bool:
        """Main agent loop: plan → code → test → fix (max 3 attempts)."""
        print(f'\n[Agent] Starting AI Agent for {self.bank.upper()} parser...\n')
        
        # Step 1: Analyze inputs
        schema = self._analyze_schema()
        input_sample = FileAnalyzer.extract_sample(self.input_path, self.input_format)
        
        # Step 2: Generate initial parser
        print('[Agent] 🤖 Generating initial parser with LLM...')
        parser_code = self._generate_parser(schema, input_sample, attempt=1)
        self._save_parser(parser_code)
        
        # Step 3: Test and self-fix loop
        for attempt in range(1, 4):
            print(f'\n[Agent] 🧪 Test attempt {attempt}/3...')
            success, error_msg = self._test_parser()
            
            if success:
                print('\n[Agent] ✅ Tests passed! Parser working correctly.')
                return True
            
            if attempt < 3:
                print(f'[Agent] ❌ Test failed: {error_msg}')
                print(f'[Agent] 🔧 Asking LLM to fix the issue...')
                parser_code = self._fix_parser(parser_code, error_msg, input_sample, attempt + 1)
                self._save_parser(parser_code)
            else:
                print(f'\n[Agent] ❌ Failed after 3 attempts.')
                print(f'Last error: {error_msg}')
        
        return False
    
    def _analyze_schema(self) -> dict:
        """Analyze expected CSV to understand output schema."""
        df = pd.read_csv(self.expected_csv)
        return {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_rows': df.head(3).to_dict('records'),
            'row_count': len(df),
            'has_dates': any('date' in col.lower() for col in df.columns),
            'has_amounts': any(col.lower() in ['amount', 'debit', 'credit', 'balance'] 
                              for col in df.columns)
        }
    
    def _generate_parser(self, schema: dict, input_sample: dict, attempt: int) -> str:
        """Use LLM to generate parser code based on input format."""
        
        # Build format-specific instructions
        if self.input_format == 'pdf':
            format_instructions = """
INPUT FORMAT: PDF
- Use pdfplumber library to read PDF
- Extract text with page.extract_text()
- Parse transaction rows (look for date patterns)
- Handle multi-page PDFs
- Clean and split text into structured data

Example code structure:
```python
import pdfplumber
def parse(file_path: str) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Parse text into rows
    return pd.DataFrame(rows, columns=expected_columns)
```
"""
        elif self.input_format == 'excel':
            format_instructions = """
INPUT FORMAT: EXCEL (XLSX/XLS)
- Use pandas.read_excel() to read Excel file
- May have multiple sheets - check all sheets or use sheet_name parameter
- Handle merged cells and formatting
- Clean column names and data

Example code structure:
```python
def parse(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # OR if specific sheet: df = pd.read_excel(file_path, sheet_name='Sheet1')
    # Clean and transform data
    return df[expected_columns]
```
"""
        else:  # CSV
            format_instructions = """
INPUT FORMAT: CSV
- Use pandas.read_csv() to read CSV file
- Handle different encodings (utf-8, latin-1)
- Handle different delimiters if needed
- Clean column names

Example code structure:
```python
def parse(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Clean and transform data
    return df[expected_columns]
```
"""
        
        prompt = f"""You are an expert Python developer. Generate a bank statement parser.

REQUIREMENTS:
1. Function signature: parse(file_path: str) -> pd.DataFrame
2. Input file format: {self.input_format.upper()}
3. Output MUST be CSV-ready DataFrame with this EXACT schema:
   Columns: {schema['columns']}
   Expected rows: ~{schema['row_count']}

4. Expected output sample:
{schema['sample_rows']}

5. Input file sample:
{input_sample}

{format_instructions}

CRITICAL RULES:
- Output DataFrame must have EXACTLY these columns: {schema['columns']}
- All column names must match exactly (case-sensitive)
- Handle missing/null values appropriately
- Convert data types correctly (dates, numbers, strings)
- Return clean, validated DataFrame ready for CSV export
- Include error handling
- Add docstring explaining the function

Generate ONLY the complete Python code for custom_parsers/{self.bank}_parser.py
Include all necessary imports at the top.
"""
        
        response = self.llm.generate(prompt)
        code = self._extract_code(response)
        self.history.append({
            "attempt": attempt, 
            "prompt": prompt, 
            "code": code,
            "input_format": self.input_format
        })
        return code
    
    def _fix_parser(self, current_code: str, error_msg: str, input_sample: dict, attempt: int) -> str:
        """Use LLM to fix parser based on error."""
        prompt = f"""The parser has failed. Please analyze and fix the issue.

INPUT FORMAT: {self.input_format.upper()}
CURRENT CODE:
```python
{current_code}
```

ERROR MESSAGE:
{error_msg}

INPUT FILE SAMPLE:
{input_sample}

REQUIRED OUTPUT SCHEMA:
Columns: {self._analyze_schema()['columns']}

INSTRUCTIONS:
1. Analyze the error carefully
2. Check if the parsing logic matches the input format
3. Ensure output DataFrame has correct columns
4. Fix data type conversions
5. Handle edge cases

Provide the COMPLETE FIXED Python code (not just the changes).
"""
        
        response = self.llm.generate(prompt)
        code = self._extract_code(response)
        self.history.append({
            "attempt": attempt, 
            "prompt": prompt, 
            "code": code, 
            "error": error_msg
        })
        return code
    
    def _extract_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response."""
        code = llm_response
        
        # Remove markdown code blocks
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            parts = code.split('```')
            # Find the part that looks like Python code
            for part in parts:
                if 'import' in part or 'def parse' in part:
                    code = part
                    break
        
        return code.strip()
    
    def _save_parser(self, code: str):
        """Save parser code to file."""
        self.parser_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure proper imports based on format
        if self.input_format == 'pdf' and 'import pdfplumber' not in code:
            code = 'import pdfplumber\n' + code
        
        self.parser_path.write_text(code, encoding='utf-8')
        print(f'[Agent] 💾 Saved parser to {self.parser_path}')
    
    def _test_parser(self) -> tuple[bool, Optional[str]]:
        """Test the parser against expected CSV output."""
        try:
            # Import the parser dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"{self.bank}_parser", 
                self.parser_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Run parser
            result_df = module.parse(str(self.input_path))
            expected_df = pd.read_csv(self.expected_csv)
            
            # Validation checks
            # 1. Check if result is a DataFrame
            if not isinstance(result_df, pd.DataFrame):
                return False, f"Parser didn't return a DataFrame, got {type(result_df)}"
            
            # 2. Check schema (column names)
            if list(result_df.columns) != list(expected_df.columns):
                return False, f"Column mismatch:\nGot: {list(result_df.columns)}\nExpected: {list(expected_df.columns)}"
            
            # 3. Check row count (allow ±10% variance)
            row_diff = abs(len(result_df) - len(expected_df))
            if row_diff > len(expected_df) * 0.1:
                return False, f"Row count mismatch: got {len(result_df)} rows, expected {len(expected_df)} rows (diff: {row_diff})"
            
            # 4. Check if DataFrame is empty when it shouldn't be
            if len(expected_df) > 0 and len(result_df) == 0:
                return False, "Parser returned empty DataFrame"
            
            # 5. Test CSV export (ensure it's CSV-ready)
            try:
                csv_str = result_df.to_csv(index=False)
                if not csv_str:
                    return False, "DataFrame cannot be converted to CSV"
            except Exception as e:
                return False, f"CSV export failed: {str(e)}"
            
            # 6. Check data equality (exact match)
            if result_df.equals(expected_df):
                return True, None
            
            # 7. If not exact match, provide detailed feedback
            return False, (f"Data mismatch: parsed {len(result_df)} rows, "
                          f"but values don't match expected output. "
                          f"Check data types and value formatting.")
            
        except ImportError as e:
            return False, f"Import error: {str(e)}. Check if all required libraries are imported."
        except AttributeError as e:
            return False, f"Missing parse function: {str(e)}"
        except Exception as e:
            return False, f"Runtime error: {type(e).__name__}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='AI Agent for bank statement parsing (PDF/Excel/CSV → CSV)'
    )
    parser.add_argument('--bank', default='icici', 
                       help='Bank name (e.g., icici, sbi, hdfc)')
    parser.add_argument('--input', default='icici sample.pdf', 
                       help='Input file (PDF, XLSX, XLS, or CSV)')
    parser.add_argument('--expected', default='result.csv', 
                       help='Expected output CSV')
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    expected_csv = Path(args.expected)
    
    if not input_path.exists():
        print(f'❌ Error: Input file not found: {input_path}')
        sys.exit(1)
    
    if not expected_csv.exists():
        print(f'❌ Error: Expected CSV not found: {expected_csv}')
        sys.exit(1)
    
    # Check if custom_parsers directory exists
    Path('custom_parsers').mkdir(exist_ok=True)
    init_file = Path('custom_parsers/__init__.py')
    if not init_file.exists():
        init_file.touch()
    
    # Run agent
    try:
        agent = AgentAsCoder(args.bank, input_path, expected_csv)
        success = agent.run()
        
        if success:
            print(f'\n✅ SUCCESS! Parser saved to: custom_parsers/{args.bank}_parser.py')
            print(f'You can now use: from custom_parsers.{args.bank}_parser import parse')
        
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n❌ Fatal error: {type(e).__name__}: {str(e)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
