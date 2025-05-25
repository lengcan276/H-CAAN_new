import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from pathlib import Path
import tempfile
import json
import zipfile
import time

class FileUploader:
    """
    Enhanced file uploader component for H-CAAN with specialized molecular data handling
    capabilities, support for large files, batch processing, and preview functionality.
    """
    
    # Supported file formats
    MOLECULE_FORMATS = {
        'sdf': 'Structure-Data File (SDF)',
        'mol': 'MDL Molfile',
        'pdb': 'Protein Data Bank (PDB)',
        'smi': 'SMILES',
        'csv': 'CSV with SMILES/InChI column',
        'txt': 'Text file with SMILES/InChI',
        'xlsx': 'Excel with SMILES/InChI column',
        'json': 'JSON with molecular data'
    }
    
    # Feature extraction formats
    FEATURE_FORMATS = {
        'npy': 'NumPy arrays',
        'npz': 'NumPy compressed arrays',
        'csv': 'CSV feature matrices',
        'pkl': 'Pickle files',
        'h5': 'HDF5 files'
    }
    
    @staticmethod
    def upload_molecules(key="molecule_upload", allowed_formats=None, 
                        instructions=None, show_preview=True, 
                        max_file_size=200, process_immediately=False):
        """
        Upload molecular data files with preview and validation
        
        Args:
            key: Unique key for the uploader
            allowed_formats: List of allowed file formats (None=all supported)
            instructions: Custom instructions to display
            show_preview: Whether to show data preview
            max_file_size: Maximum file size in MB
            process_immediately: Whether to process data upon upload
            
        Returns:
            Dict containing the processed data or None if not processed
        """
        if allowed_formats is None:
            allowed_formats = list(FileUploader.MOLECULE_FORMATS.keys())
            
        allowed_formats_str = ", ".join([f".{fmt}" for fmt in allowed_formats])
        
        # Create instructions
        if instructions is None:
            instructions = (
                f"Upload molecular data in any of these formats: {allowed_formats_str}. "
                f"Maximum file size: {max_file_size}MB."
            )
        
        st.write(instructions)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=allowed_formats, 
            key=key,
            help="Select a file containing molecular structures to analyze"
        )
        
        if uploaded_file is None:
            return None
        
        # File validation
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_file_size:
            st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum allowed size ({max_file_size}MB)")
            return None
        
        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # File info
        st.success(f"File '{uploaded_file.name}' uploaded successfully ({file_size_mb:.2f}MB)")
        
        # Process the file
        if process_immediately or st.button("Process File", key=f"{key}_process"):
            with st.spinner("Processing molecular data..."):
                try:
                    data = FileUploader._process_molecular_file(uploaded_file, file_ext)
                    
                    if show_preview:
                        FileUploader._show_molecular_preview(data, file_ext)
                        
                    return data
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.error("Please check that the file format is correct and contains valid molecular data.")
                    return None
                
        return None
    
    @staticmethod
    def upload_features(key="feature_upload", allowed_formats=None, 
                       instructions=None, show_preview=True, 
                       max_file_size=500, process_immediately=False):
        """
        Upload pre-computed molecular features or embeddings
        
        Args:
            key: Unique key for the uploader
            allowed_formats: List of allowed file formats (None=all supported)
            instructions: Custom instructions to display
            show_preview: Whether to show data preview
            max_file_size: Maximum file size in MB
            process_immediately: Whether to process data upon upload
            
        Returns:
            Dict containing the processed features or None if not processed
        """
        if allowed_formats is None:
            allowed_formats = list(FileUploader.FEATURE_FORMATS.keys())
            
        allowed_formats_str = ", ".join([f".{fmt}" for fmt in allowed_formats])
        
        # Create instructions
        if instructions is None:
            instructions = (
                f"Upload pre-computed molecular features in any of these formats: {allowed_formats_str}. "
                f"Maximum file size: {max_file_size}MB."
            )
        
        st.write(instructions)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=allowed_formats, 
            key=key,
            help="Select a file containing molecular features"
        )
        
        if uploaded_file is None:
            return None
        
        # File validation
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_file_size:
            st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum allowed size ({max_file_size}MB)")
            return None
        
        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # File info
        st.success(f"File '{uploaded_file.name}' uploaded successfully ({file_size_mb:.2f}MB)")
        
        # Process the file
        if process_immediately or st.button("Process File", key=f"{key}_process"):
            with st.spinner("Processing feature data..."):
                try:
                    data = FileUploader._process_feature_file(uploaded_file, file_ext)
                    
                    if show_preview and hasattr(data, 'shape'):
                        st.write(f"Feature shape: {data.shape}")
                        
                        if show_preview and len(data.shape) == 2 and data.shape[0] < 1000:
                            preview_df = pd.DataFrame(
                                data[:10, :10] if data.shape[1] > 10 else data[:10, :]
                            )
                            st.write("Feature preview (first 10 rows, up to 10 columns):")
                            st.dataframe(preview_df)
                        
                    return data
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    return None
                
        return None
    
    @staticmethod
    def batch_upload_directory(key="batch_upload", allowed_formats=None, 
                             max_files=100, max_total_size=1000, 
                             show_progress=True):
        """
        Simulate uploading a directory of files through a zip archive
        
        Args:
            key: Unique key for the uploader
            allowed_formats: List of allowed file formats (None=all supported)
            max_files: Maximum number of files
            max_total_size: Maximum total size in MB
            show_progress: Whether to show a progress bar
            
        Returns:
            Dict containing processed files or None if not processed
        """
        if allowed_formats is None:
            allowed_formats = list(FileUploader.MOLECULE_FORMATS.keys())
            
        allowed_formats_str = ", ".join([f".{fmt}" for fmt in allowed_formats])
        
        st.write(
            f"Upload a ZIP file containing multiple molecular data files in any of these formats: "
            f"{allowed_formats_str}. Maximum total size: {max_total_size}MB."
        )
        
        # File upload
        uploaded_zip = st.file_uploader(
            "Choose a ZIP file", 
            type=["zip"], 
            key=key,
            help="Select a ZIP archive containing multiple molecular structure files"
        )
        
        if uploaded_zip is None:
            return None
        
        # File validation
        file_size_mb = uploaded_zip.size / (1024 * 1024)
        if file_size_mb > max_total_size:
            st.error(f"File size ({file_size_mb:.1f}MB) exceeds the maximum allowed size ({max_total_size}MB)")
            return None
        
        # File info
        st.success(f"ZIP file '{uploaded_zip.name}' uploaded successfully ({file_size_mb:.2f}MB)")
        
        # Process the ZIP file
        if st.button("Extract and Process Files", key=f"{key}_process"):
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP contents
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Scan for valid files
                valid_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_ext = file.split('.')[-1].lower()
                        if file_ext in allowed_formats:
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, temp_dir)
                            file_size = os.path.getsize(file_path) / (1024 * 1024)
                            valid_files.append((relative_path, file_path, file_ext, file_size))
                
                # Check if we have too many files
                if len(valid_files) > max_files:
                    st.error(f"Too many files found ({len(valid_files)}). Maximum allowed: {max_files}")
                    return None
                
                if len(valid_files) == 0:
                    st.error(f"No valid files found in the ZIP archive. Supported formats: {allowed_formats_str}")
                    return None
                
                # Process files
                st.write(f"Found {len(valid_files)} valid files to process")
                
                processed_data = {}
                
                progress_container = st.container()
                if show_progress:
                    progress_bar = progress_container.progress(0)
                
                for i, (rel_path, file_path, file_ext, file_size) in enumerate(valid_files):
                    try:
                        # Update progress
                        if show_progress:
                            progress_bar.progress((i + 1) / len(valid_files))
                            progress_container.write(f"Processing: {rel_path} ({file_size:.2f}MB)")
                        
                        # Open and process the file
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = os.path.basename(file_path)
                            
                            data = FileUploader._process_molecular_file(file_obj, file_ext)
                            processed_data[rel_path] = data
                    
                    except Exception as e:
                        st.error(f"Error processing {rel_path}: {str(e)}")
                
                if show_progress:
                    progress_bar.progress(1.0)
                
                # Summary
                st.success(f"Successfully processed {len(processed_data)} out of {len(valid_files)} files")
                
                return processed_data
                
        return None
    
    @staticmethod
    def upload_custom_datasets(key="dataset_upload", dataset_schema=None):
        """
        Upload a custom dataset with flexible schema definition
        
        Args:
            key: Unique key for the uploader
            dataset_schema: Optional schema definition for the dataset
            
        Returns:
            Dict with processed dataset
        """
        st.write("Upload a custom dataset for molecular property prediction")
        
        # Template download
        st.markdown("""
        ### Dataset Format Requirements
        
        Your dataset should include:
        1. A column containing molecular structures (SMILES, InChI, or other identifiers)
        2. Target property columns for prediction
        3. Optional: Additional metadata columns
        
        Use the template below as a starting point:
        """)
        
        # Create template file
        template_df = pd.DataFrame({
            'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1'],
            'molecule_name': ['Ethanol', 'Acetic acid', 'Benzene'],
            'property_1': [0.5, 1.2, 2.3],
            'property_2': [True, False, True],
            'metadata': ['solvent', 'reagent', 'building block']
        })
        
        # Create download button for template
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Template CSV",
            data=csv_str,
            file_name="template_dataset.csv",
            mime="text/csv",
            key=f"{key}_template"
        )
        
        # Upload interface
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=["csv", "xlsx", "tsv", "txt"], 
            key=key
        )
        
        if uploaded_file is None:
            return None
        
        # Process the file
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_ext in ['tsv', 'txt']:
                df = pd.read_csv(uploaded_file, sep='\t')
            
            # Dataset overview
            st.success(f"Dataset loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Preview the data
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Column selection for structure and properties
            st.write("### Configure Dataset Columns")
            
            # Structure column
            struct_cols = [col for col in df.columns if any(kw in col.lower() for 
                                                         kw in ['smiles', 'inchi', 'mol', 'structure'])]
            
            if struct_cols:
                structure_col = st.selectbox(
                    "Select the column containing molecular structures",
                    options=struct_cols + [col for col in df.columns if col not in struct_cols],
                    index=0 if struct_cols else 0,
                    key=f"{key}_struct_col"
                )
            else:
                structure_col = st.selectbox(
                    "Select the column containing molecular structures",
                    options=df.columns,
                    key=f"{key}_struct_col"
                )
            
            # Property columns
            prop_cols = [col for col in df.columns if any(kw in col.lower() for 
                                                       kw in ['property', 'target', 'value', 'activity'])]
            
            if prop_cols:
                property_cols = st.multiselect(
                    "Select the target property columns",
                    options=[col for col in df.columns if col != structure_col],
                    default=prop_cols,
                    key=f"{key}_prop_cols"
                )
            else:
                property_cols = st.multiselect(
                    "Select the target property columns",
                    options=[col for col in df.columns if col != structure_col],
                    key=f"{key}_prop_cols"
                )
                
            if not property_cols:
                st.warning("Please select at least one property column")
            
            # Process the dataset
            if st.button("Process Dataset", key=f"{key}_process"):
                with st.spinner("Processing dataset..."):
                    # Validate molecular structures
                    valid_count = 0
                    invalid_indices = []
                    
                    with st.expander("Structure Validation"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, struct in enumerate(df[structure_col]):
                            try:
                                # Try to parse as SMILES
                                mol = Chem.MolFromSmiles(str(struct))
                                if mol is None:
                                    # Try to parse as InChI if SMILES fails
                                    mol = Chem.MolFromInchi(str(struct))
                                
                                if mol is None:
                                    invalid_indices.append(i)
                                else:
                                    valid_count += 1
                                    
                            except:
                                invalid_indices.append(i)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))
                            if i % 100 == 0 or i == len(df) - 1:
                                status_text.write(f"Processed {i+1}/{len(df)} structures")
                    
                    # Report validation results
                    if invalid_indices:
                        st.warning(f"Found {len(invalid_indices)} invalid molecular structures ({valid_count} valid)")
                        st.write("Preview of invalid entries:")
                        st.dataframe(df.iloc[invalid_indices[:10]] if len(invalid_indices) > 10 
                                   else df.iloc[invalid_indices])
                    else:
                        st.success(f"All {valid_count} molecular structures are valid")
                    
                    # Create processed dataset
                    processed_data = {
                        'dataframe': df,
                        'structure_column': structure_col,
                        'property_columns': property_cols,
                        'valid_indices': [i for i in range(len(df)) if i not in invalid_indices],
                        'invalid_indices': invalid_indices
                    }
                    
                    return processed_data
            
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            return None
        
        return None
    
    @staticmethod
    def _process_molecular_file(file_obj, file_ext):
        """Process different molecular file formats"""
        if file_ext == 'sdf':
            # Use RDKit to read SDF file
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            
            # Read SDF and convert to DataFrame
            sdf_data = list(Chem.SDMolSupplier(tmp_path))
            os.unlink(tmp_path)  # Clean up temp file
            
            # Convert molecules to DataFrame
            mols_df = pd.DataFrame({'ROMol': sdf_data})
            
            # Extract properties
            if len(sdf_data) > 0 and sdf_data[0] is not None:
                prop_keys = list(sdf_data[0].GetPropsAsDict().keys())
                for key in prop_keys:
                    mols_df[key] = [mol.GetProp(key) if mol and mol.HasProp(key) else None 
                                   for mol in sdf_data]
            
            # Add SMILES column
            mols_df['SMILES'] = [Chem.MolToSmiles(mol) if mol else None for mol in sdf_data]
            
            return {
                'dataframe': mols_df,
                'molecules': sdf_data,
                'source_format': 'sdf'
            }
            
        elif file_ext == 'mol':
            # Read single MOL file
            mol_data = file_obj.read()
            mol = Chem.MolFromMolBlock(mol_data.decode('utf-8'))
            
            if mol:
                # Convert to DataFrame with a single row
                mols_df = pd.DataFrame({
                    'ROMol': [mol],
                    'SMILES': [Chem.MolToSmiles(mol)]
                })
                
                return {
                    'dataframe': mols_df,
                    'molecules': [mol],
                    'source_format': 'mol'
                }
            else:
                raise ValueError("Could not parse MOL file")
                
        elif file_ext == 'pdb':
            # Read PDB file
            pdb_data = file_obj.read().decode('utf-8')
            mol = Chem.MolFromPDBBlock(pdb_data)
            
            if mol:
                # Convert to DataFrame with a single row
                mols_df = pd.DataFrame({
                    'ROMol': [mol],
                    'SMILES': [Chem.MolToSmiles(mol)]
                })
                
                return {
                    'dataframe': mols_df,
                    'molecules': [mol],
                    'pdb_block': pdb_data,
                    'source_format': 'pdb'
                }
            else:
                raise ValueError("Could not parse PDB file")
                
        elif file_ext == 'smi':
            # Read SMILES file
            smiles_data = file_obj.read().decode('utf-8').splitlines()
            
            # Parse SMILES lines (handle different formats)
            molecules = []
            smiles_list = []
            names = []
            
            for line in smiles_data:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if parts:
                        smiles = parts[0]
                        name = parts[1] if len(parts) > 1 else f"Mol_{len(smiles_list)+1}"
                        
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            molecules.append(mol)
                            smiles_list.append(smiles)
                            names.append(name)
            
            # Convert to DataFrame
            mols_df = pd.DataFrame({
                'ROMol': molecules,
                'SMILES': smiles_list,
                'Name': names
            })
            
            return {
                'dataframe': mols_df,
                'molecules': molecules,
                'source_format': 'smi'
            }
            
        elif file_ext == 'csv':
            # Read CSV file
            df = pd.read_csv(file_obj)
            
            # Find SMILES or InChI column
            smiles_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ['smiles', 'smi', 'inchi', 'structure']):
                    smiles_col = col
                    break
            
            if smiles_col is None and len(df.columns) > 0:
                # Try the first column if no obvious SMILES column is found
                smiles_col = df.columns[0]
            
            if smiles_col:
                # Convert SMILES to molecules
                molecules = []
                valid_indices = []
                
                for i, smiles in enumerate(df[smiles_col]):
                    try:
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol:
                            molecules.append(mol)
                            valid_indices.append(i)
                        else:
                            # Try InChI if SMILES parsing fails
                            mol = Chem.MolFromInchi(str(smiles))
                            if mol:
                                molecules.append(mol)
                                valid_indices.append(i)
                    except:
                        # Skip invalid entries
                        pass
                
                # Create new DataFrame with only valid molecules
                if molecules:
                    valid_df = df.iloc[valid_indices].copy()
                    valid_df['ROMol'] = molecules
                    
                    return {
                        'dataframe': valid_df,
                        'molecules': molecules,
                        'original_df': df,
                        'smiles_column': smiles_col,
                        'source_format': 'csv'
                    }
                else:
                    raise ValueError(f"No valid molecules found in column '{smiles_col}'")
            else:
                raise ValueError("Could not identify a SMILES or InChI column")
                
        elif file_ext == 'xlsx':
            # Read Excel file
            xls = pd.ExcelFile(file_obj)
            sheet_name = xls.sheet_names[0]  # Use the first sheet
            
            df = pd.read_excel(file_obj, sheet_name=sheet_name)
            
            # Find SMILES or InChI column
            smiles_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ['smiles', 'smi', 'inchi', 'structure']):
                    smiles_col = col
                    break
            
            if smiles_col is None and len(df.columns) > 0:
                # Try the first column if no obvious SMILES column is found
                smiles_col = df.columns[0]
            
            if smiles_col:
                # Convert SMILES to molecules
                molecules = []
                valid_indices = []
                
                for i, smiles in enumerate(df[smiles_col]):
                    try:
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol:
                            molecules.append(mol)
                            valid_indices.append(i)
                        else:
                            # Try InChI if SMILES parsing fails
                            mol = Chem.MolFromInchi(str(smiles))
                            if mol:
                                molecules.append(mol)
                                valid_indices.append(i)
                    except:
                        # Skip invalid entries
                        pass
                
                # Create new DataFrame with only valid molecules
                if molecules:
                    valid_df = df.iloc[valid_indices].copy()
                    valid_df['ROMol'] = molecules
                    
                    return {
                        'dataframe': valid_df,
                        'molecules': molecules,
                        'original_df': df,
                        'smiles_column': smiles_col,
                        'source_format': 'xlsx',
                        'sheet_name': sheet_name
                    }
                else:
                    raise ValueError(f"No valid molecules found in column '{smiles_col}'")
            else:
                raise ValueError("Could not identify a SMILES or InChI column")
                
        elif file_ext == 'json':
            # Read JSON file
            json_data = json.loads(file_obj.read().decode('utf-8'))
            
            if isinstance(json_data, list):
                # List of molecules
                molecules = []
                smiles_list = []
                properties = []
                
                for item in json_data:
                    if isinstance(item, str):
                        # Simple list of SMILES
                        smiles = item
                        props = {}
                    elif isinstance(item, dict) and 'smiles' in item:
                        # Dictionary with SMILES and properties
                        smiles = item['smiles']
                        props = {k: v for k, v in item.items() if k != 'smiles'}
                    elif isinstance(item, dict) and 'structure' in item:
                        # Dictionary with structure and properties
                        smiles = item['structure']
                        props = {k: v for k, v in item.items() if k != 'structure'}
                    else:
                        # Skip invalid items
                        continue
                    
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        molecules.append(mol)
                        smiles_list.append(smiles)
                        properties.append(props)
                
                # Convert to DataFrame
                if molecules:
                    mols_df = pd.DataFrame({
                        'ROMol': molecules,
                        'SMILES': smiles_list
                    })
                    
                    # Add properties as columns
                    for i, props in enumerate(properties):
                        for k, v in props.items():
                            if k not in mols_df:
                                mols_df[k] = None
                            mols_df.at[i, k] = v
                    
                    return {
                        'dataframe': mols_df,
                        'molecules': molecules,
                        'source_format': 'json'
                    }
                else:
                    raise ValueError("No valid molecules found in JSON file")
                    
            elif isinstance(json_data, dict) and 'molecules' in json_data:
                # Dictionary with 'molecules' key
                mol_list = json_data['molecules']
                if isinstance(mol_list, list):
                    # Process the list of molecules
                    molecules = []
                    smiles_list = []
                    properties = []
                    
                    for item in mol_list:
                        if isinstance(item, str):
                            smiles = item
                            props = {}
                        elif isinstance(item, dict) and ('smiles' in item or 'structure' in item):
                            smiles = item.get('smiles', item.get('structure'))
                            props = {k: v for k, v in item.items() if k not in ['smiles', 'structure']}
                        else:
                            continue
                        
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            molecules.append(mol)
                            smiles_list.append(smiles)
                            properties.append(props)
                    
                    # Convert to DataFrame
                    if molecules:
                        mols_df = pd.DataFrame({
                            'ROMol': molecules,
                            'SMILES': smiles_list
                        })
                        
                        # Add properties as columns
                        for i, props in enumerate(properties):
                            for k, v in props.items():
                                if k not in mols_df:
                                    mols_df[k] = None
                                mols_df.at[i, k] = v
                        
                        return {
                            'dataframe': mols_df,
                            'molecules': molecules,
                            'metadata': {k: v for k, v in json_data.items() if k != 'molecules'},
                            'source_format': 'json'
                        }
                    else:
                        raise ValueError("No valid molecules found in JSON file")
                else:
                    raise ValueError("Invalid JSON format: 'molecules' key should contain a list")
            else:
                raise ValueError("Invalid JSON format")
                
        elif file_ext == 'txt':
            # Try to detect format based on content
            content = file_obj.read().decode('utf-8')
            
            # Check if it's a SMILES file
            lines = content.splitlines()
            if lines and all(Chem.MolFromSmiles(line.split()[0]) is not None 
                           for line in lines[:5] if line.strip() and not line.startswith('#')):
                # Rewind file and process as SMILES
                file_obj.seek(0)
                return FileUploader._process_molecular_file(file_obj, 'smi')
            
            # Check if it's a MOL file
            if "V2000" in content or "V3000" in content:
                # Process as MOL
                mol = Chem.MolFromMolBlock(content)
                if mol:
                    # Convert to DataFrame with a single row
                    mols_df = pd.DataFrame({
                        'ROMol': [mol],
                        'SMILES': [Chem.MolToSmiles(mol)]
                    })
                    
                    return {
                        'dataframe': mols_df,
                        'molecules': [mol],
                        'source_format': 'mol'
                    }
            
            # Check if it's a PDB file
            if "HEADER" in content and "ATOM" in content:
                # Process as PDB
                mol = Chem.MolFromPDBBlock(content)
                if mol:
                    # Convert to DataFrame with a single row
                    mols_df = pd.DataFrame({
                        'ROMol': [mol],
                        'SMILES': [Chem.MolToSmiles(mol)]
                    })
                    
                    return {
                        'dataframe': mols_df,
                        'molecules': [mol],
                        'pdb_block': content,
                        'source_format': 'pdb'
                    }
            
            # Check if it's a CSV/TSV file
            if "\t" in lines[0]:
                # Try to process as TSV
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep='\t')
                file_obj.seek(0)
                return FileUploader._process_molecular_file(io.BytesIO(df.to_csv().encode()), 'csv')
            elif "," in lines[0]:
                # Try to process as CSV
                file_obj.seek(0)
                return FileUploader._process_molecular_file(file_obj, 'csv')
            
            raise ValueError("Could not determine the format of the text file")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def _process_feature_file(file_obj, file_ext):
        """Process different feature file formats"""
        if file_ext == 'npy':
            # NumPy array
            return np.load(file_obj)
            
        elif file_ext == 'npz':
            # Compressed NumPy arrays
            return np.load(file_obj)
            
        elif file_ext == 'csv':
            # CSV feature matrix
            return pd.read_csv(file_obj).values
            
        elif file_ext == 'pkl' or file_ext == 'pickle':
            # Pickle file
            import pickle
            return pickle.load(file_obj)
            
        elif file_ext == 'h5' or file_ext == 'hdf5':
            # HDF5 file
            import h5py
            with h5py.File(file_obj, 'r') as f:
                # Return the first dataset found
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        return f[k][()]
            
            raise ValueError("No dataset found in HDF5 file")
            
        else:
            raise ValueError(f"Unsupported feature file format: {file_ext}")
    
    @staticmethod
    def _show_molecular_preview(data, file_ext):
        """Show a preview of the molecular data"""
        if 'dataframe' in data:
            df = data['dataframe']
            
            # Show basic stats
            st.write(f"Dataset contains {len(df)} molecules")
            
            # Show column info
            st.write("Columns:", ", ".join(df.columns))
            
            # Preview the DataFrame
            preview_df = df.drop('ROMol', axis=1, errors='ignore').head(5)
            st.write("Data preview:")
            st.dataframe(preview_df)
            
            # Show a few molecules
            from streamlit.components.v1 import html
            from rdkit.Chem import Draw
            
            st.write("Molecule preview:")
            
            # Get up to 5 molecules
            preview_mols = data['molecules'][:5] if len(data['molecules']) > 5 else data['molecules']
            
            # Create a grid image
            if preview_mols:
                img = Draw.MolsToGridImage(
                    preview_mols,
                    molsPerRow=min(3, len(preview_mols)),
                    subImgSize=(200, 200),
                    legends=[f"Molecule {i+1}" for i in range(len(preview_mols))]
                )
                
                # Display the image
                st.image(img)