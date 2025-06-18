import sys
import os
import re
import json
from pathlib import Path
from custom_loaders import CSVLoader, ExcelLoader, EnhancedExcelLoader

def test_csv_loader():
    print("\n" + "=" * 50)
    print("=== Testing CSV Loader ===")
    print("=" * 50)
    
    csv_path = "test_files/test_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Test file not found at {csv_path}")
        return
    
    print(f"Loading CSV file: {csv_path}")
    
    # Suppress pandas warnings during testing
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = CSVLoader(csv_path)
        documents = loader.load()
    
    print(f"Number of documents created: {len(documents)}")
    print(f"Document types: {[doc.metadata.get('type') for doc in documents]}")
    
    # Test full document with enhanced formatting
    full_doc = documents[0]
    print("\nFull CSV Document:")
    print(f"- Content length: {len(full_doc.page_content)} chars")
    has_table = bool(re.search(r'\|\s+\|', full_doc.page_content))
    print(f"- Has markdown table: {has_table}")
    print(f"- Has HTML table: {'html_table' in full_doc.metadata}")
    
    # Test chart detection metadata
    print("\nChart Detection Metadata:")
    print(f"- Numeric columns: {full_doc.metadata.get('numeric_columns', 'None')}")
    print(f"- Date columns: {full_doc.metadata.get('date_columns', 'None')}")
    print(f"- Chart potential: {full_doc.metadata.get('has_chart_potential', False)}")
    
    # Test statistics in metadata
    print("\nStatistics in Metadata:")
    stats_keys = [k for k in full_doc.metadata.keys() if k.startswith('stats_')]
    for key in stats_keys:
        print(f"- {key}: {full_doc.metadata[key]}")
    
    # Test row documents
    if len(documents) > 1:
        row_doc = documents[1]
        print("\n----------------------------------------")
        print("Row Document:")
        print("----------------------------------------")
        print(f"- Content: {row_doc.page_content[:100]}...")
        print(f"- Metadata: {json.dumps({k: v for k, v in row_doc.metadata.items() if k.startswith('numeric_')}, indent=2)}")

def test_excel_loader():
    print("\n" + "=" * 50)
    print("=== Testing Excel Loader ===")
    print("=" * 50)
    
    excel_path = "test_files/test_data.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: Test file not found at {excel_path}")
        return
    
    print(f"Loading Excel file: {excel_path}")
    
    # Suppress pandas warnings during testing
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = ExcelLoader(excel_path)
        documents = loader.load()
    
    print(f"Number of documents created: {len(documents)}")
    
    # Group documents by sheet
    sheet_docs = {}
    for doc in documents:
        sheet = doc.metadata.get('sheet', 'unknown')
        doc_type = doc.metadata.get('type', '')
        
        if sheet not in sheet_docs:
            sheet_docs[sheet] = {'sheet_docs': [], 'row_docs': []}
            
        if doc_type == 'excel_sheet':
            sheet_docs[sheet]['sheet_docs'].append(doc)
        elif doc_type == 'excel_row':
            sheet_docs[sheet]['row_docs'].append(doc)
    
    # Test each sheet's documents
    for sheet, docs in sheet_docs.items():
        print(f"\nSheet: {sheet}")
        
        # Test sheet document with enhanced formatting
        if docs['sheet_docs']:
            sheet_doc = docs['sheet_docs'][0]
            print("\n  Sheet Document:")
            print(f"  - Content length: {len(sheet_doc.page_content)} chars")
            has_table = bool(re.search(r'\|\s+\|', sheet_doc.page_content))
            print(f"  - Has markdown table: {has_table}")
            print(f"  - Has HTML table: {'html_table' in sheet_doc.metadata}")
            
            # Test chart detection metadata
            print("\n  Chart Detection Metadata:")
            print(f"  - Numeric columns: {sheet_doc.metadata.get('numeric_columns', 'None')}")
            print(f"  - Date columns: {sheet_doc.metadata.get('date_columns', 'None')}")
            print(f"  - Chart potential: {sheet_doc.metadata.get('has_chart_potential', False)}")
            
            # Test statistics in metadata
            print("\n  Statistics in Metadata:")
            stats_keys = [k for k in sheet_doc.metadata.keys() if k.startswith('stats_')]
            for key in stats_keys:
                print(f"  - {key}: {sheet_doc.metadata[key]}")
        
        # Test row documents
        if docs['row_docs']:
            row_doc = docs['row_docs'][0]
            print("\n  Row Document:")
            print(f"  - Content: {row_doc.page_content[:100]}...")
            print(f"  - Metadata: {json.dumps({k: v for k, v in row_doc.metadata.items() if k.startswith('numeric_')}, indent=2)}")
        


def create_test_files():
    """Create test files with chart-like data for testing"""
    import pandas as pd
    import numpy as np
    import os
    
    # Ensure test directory exists
    os.makedirs("test_files", exist_ok=True)
    
    # Create a CSV with time series data (good for charts)
    dates = pd.date_range('2023-01-01', periods=10)
    data = {
        'date': dates,
        'sales': np.random.randint(100, 1000, size=10),
        'customers': np.random.randint(10, 100, size=10),
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    }
    df = pd.DataFrame(data)
    df.to_csv("test_files/chart_data.csv", index=False)
    print(f"\nCreated test CSV file: test_files/chart_data.csv")
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"CSV shape: {df.shape}")
    
    # Create Excel with multiple sheets including chart data
    with pd.ExcelWriter("test_files/chart_data.xlsx", engine='openpyxl') as writer:
        # Sheet 1: Time series data
        df.to_excel(writer, sheet_name='TimeSeries', index=False)
        
        # Sheet 2: Categorical data
        categories = ['A', 'B', 'C', 'D']
        cat_data = {
            'category': categories,
            'count': np.random.randint(50, 200, size=4),
            'percentage': np.random.uniform(0, 100, size=4)
        }
        cat_df = pd.DataFrame(cat_data)
        cat_df.to_excel(writer, sheet_name='Categories', index=False)
    
    print(f"\nCreated test Excel file: test_files/chart_data.xlsx")
    print(f"Excel sheets: TimeSeries ({df.shape}), Categories ({cat_df.shape})")

def test_chart_data():
    """Test the loaders with chart-specific data"""
    print("\n" + "=" * 50)
    print("=== Testing Chart Data Detection ===")
    print("=" * 50)
    
    # Test CSV with chart data
    csv_path = "test_files/chart_data.csv"
    if os.path.exists(csv_path):
        print(f"\nCSV Chart Data: {csv_path}")
        
        # Suppress pandas warnings during testing
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = CSVLoader(csv_path)
            documents = loader.load()
        
        print(f"Number of documents: {len(documents)}")
        
        # Find the full document
        full_docs = [doc for doc in documents if doc.metadata.get('type') == 'full_csv']
        if full_docs:
            full_doc = full_docs[0]
            print(f"- Chart potential: {full_doc.metadata.get('has_chart_potential', False)}")
            print(f"- Numeric columns: {full_doc.metadata.get('numeric_columns', 'None')}")
            print(f"- Date columns: {full_doc.metadata.get('date_columns', 'None')}")
        else:
            print("No full CSV document found")
    
    # Test Excel with chart data
    excel_path = "test_files/chart_data.xlsx"
    if os.path.exists(excel_path):
        print(f"\nExcel file: {excel_path}")
        
        # Suppress pandas warnings during testing
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = ExcelLoader(excel_path)
            documents = loader.load()
        
        print(f"Total documents: {len(documents)}")
        
        # Group by sheet and type
        sheet_docs = {}
        doc_types = set()
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            doc_types.add(doc_type)
            sheet = doc.metadata.get('sheet', 'unknown')
            
            if sheet not in sheet_docs:
                sheet_docs[sheet] = {}
            
            if doc_type not in sheet_docs[sheet]:
                sheet_docs[sheet][doc_type] = []
                
            sheet_docs[sheet][doc_type].append(doc)
        
        print(f"Document types: {doc_types}")
        print(f"Sheets found: {list(sheet_docs.keys())}")
        
        for sheet, types in sheet_docs.items():
            print(f"\n--- Excel Sheet: {sheet} ---")
            print(f"Documents in sheet: {sum(len(docs) for docs in types.values())}")
            
            sheet_doc = next((doc for doc in types.get('excel_sheet', []) if doc.metadata.get('sheet') == sheet), None)
            if sheet_doc:
                print(f"- Chart potential: {sheet_doc.metadata.get('has_chart_potential', False)}")
                print(f"- Numeric columns: {sheet_doc.metadata.get('numeric_columns', 'None')}")
                print(f"- Date columns: {sheet_doc.metadata.get('date_columns', 'None')}")
                
                # Print some statistics if available
                stats_keys = [k for k in sheet_doc.metadata.keys() if k.startswith('stats_')]
                if stats_keys:
                    print("\nStatistics:")
                    for key in sorted(stats_keys)[:6]:  # Show at most 6 stats
                        print(f"- {key}: {sheet_doc.metadata[key]}")
            else:
                print("No sheet document found")
                
            row_docs = types.get('excel_row', [])
            print(f"- Row documents: {len(row_docs)}")

def test_enhanced_excel_loader():
    print("\n" + "=" * 50)
    print("=== Testing Enhanced Excel Loader ===")
    print("=" * 50)
    
    excel_path = "test_files/test_data.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: Test file not found at {excel_path}")
        return
    
    print(f"Loading Excel file with EnhancedExcelLoader: {excel_path}")
    
    # Suppress pandas warnings during testing
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loader = EnhancedExcelLoader(
            excel_path,
            detect_merged_cells=True,
            detect_multi_level_headers=True,
            detect_tables=True,
            extract_formulas=True,
            verbose=True
        )
        documents = loader.load()
    
    print(f"Number of documents created: {len(documents)}")
    
    # Group documents by sheet and type
    sheets = {}
    doc_types = set()
    
    for doc in documents:
        sheet_name = doc.metadata.get('sheet_name', 'file_level')
        if sheet_name not in sheets:
            sheets[sheet_name] = []
        sheets[sheet_name].append(doc)
        
        # Track document types based on metadata
        if 'document_type' in doc.metadata:
            doc_types.add(doc.metadata['document_type'])
        elif sheet_name == 'file_level':
            doc_types.add('file_level')
        else:
            doc_types.add('sheet')
    
    print(f"Found {len(sheets)} sheets: {', '.join(sheets.keys())}")
    print(f"Document types: {', '.join(doc_types)}")
    
    # Examine file-level document
    if 'file_level' in sheets:
        file_doc = sheets['file_level'][0]
        print("\nFile-level document:")
        print(f"- Content: {file_doc.page_content[:100]}...")
        print(f"- Metadata: {json.dumps({k: v for k, v in file_doc.metadata.items() if not isinstance(v, list)}, indent=2)}")
    
    # Examine each sheet
    for sheet_name, docs in sheets.items():
        if sheet_name == 'file_level':
            continue
            
        print(f"\n--- Sheet: {sheet_name} ---")
        print(f"Documents in sheet: {len(docs)}")
        
        # Find sheet document (not a row document)
        sheet_docs = [d for d in docs if d.metadata.get('document_type', '') != 'excel_row']
        row_docs = [d for d in docs if d.metadata.get('document_type', '') == 'excel_row']
        
        if sheet_docs:
            sheet_doc = sheet_docs[0]
            print("\nSheet document:")
            print(f"- Content length: {len(sheet_doc.page_content)} chars")
            has_table = bool(re.search(r'\|\s+\|', sheet_doc.page_content))
            print(f"- Has markdown table: {has_table}")
            
            # Check for multi-level headers
            print(f"- Multi-level header: {sheet_doc.metadata.get('multi_level_header', False)}")
            print(f"- Header rows: {sheet_doc.metadata.get('header_rows', 0)}")
            
            # Check for table detection
            print(f"- Table count: {sheet_doc.metadata.get('table_count', 0)}")
            
            # Check for merged cells
            print(f"- Has merged cells: {sheet_doc.metadata.get('merged_cells', False)}")
            
            # Check for date columns
            print(f"- Date columns: {sheet_doc.metadata.get('date_columns', [])}")
            
            # Check for formulas
            print(f"- Has formulas: {sheet_doc.metadata.get('has_formulas', False)}")
            print(f"- Formula count: {sheet_doc.metadata.get('formula_count', 0)}")
        
        # Check row documents
        print(f"\nRow documents: {len(row_docs)}")
        if row_docs:
            sample_row = row_docs[0]
            print(f"- Sample row content: {sample_row.page_content[:100]}...")
            print(f"- Sample row metadata: {json.dumps({k: v for k, v in sample_row.metadata.items() if not isinstance(v, list)}, indent=2)}")

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#" + " " * 24 + "TESTING LOADERS" + " " * 24 + "#")
    print("#" * 70 + "\n")
    
    # Test with existing files
    test_csv_loader()
    test_excel_loader()
    test_enhanced_excel_loader()
    
    print("\n" + "#" * 70)
    print("#" + " " * 22 + "CREATING TEST FILES" + " " * 22 + "#")
    print("#" * 70 + "\n")
    
    # Create and test with chart-specific data
    create_test_files()
    
    print("\n" + "#" * 70)
    print("#" + " " * 21 + "TESTING CHART DATA" + " " * 21 + "#")
    print("#" * 70 + "\n")
    
    test_chart_data()
    
    print("\n" + "#" * 70)
    print("#" + " " * 22 + "ALL TESTS COMPLETED" + " " * 22 + "#")
    print("#" * 70 + "\n")
