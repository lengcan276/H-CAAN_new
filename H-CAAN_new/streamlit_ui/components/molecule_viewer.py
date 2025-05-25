import streamlit as st
import py3Dmol
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from io import StringIO
import base64
import re
from streamlit.components.v1 import html

class MoleculeViewer:
    """
    A Streamlit component for 3D visualization of molecules using py3Dmol.
    Supports multiple molecule formats, different visualization styles, and interactive features.
    """
    
    @staticmethod
    def _mol_to_html(mol, size=(400, 300), style='stick', surface=False, 
                    spin=False, bgcolor='white', zoom=1.0):
        """
        Convert RDKit molecule to HTML representation with py3Dmol
        
        Args:
            mol: RDKit molecule object
            size: Tuple of (width, height) for the viewer
            style: Visualization style ('stick', 'sphere', 'line', 'cartoon')
            surface: Whether to show molecular surface
            spin: Whether to enable spinning animation
            bgcolor: Background color
            zoom: Zoom level
            
        Returns:
            HTML string containing the 3D viewer
        """
        if mol is None:
            return "Invalid molecule"
        
        # Add hydrogens and generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        # Get molecule representation
        viewer = py3Dmol.view(width=size[0], height=size[1])
        molblock = Chem.MolToMolBlock(mol)
        viewer.addModel(molblock, 'mol')
        
        # Set style
        if style == 'stick':
            viewer.setStyle({'stick': {'radius': 0.15, 'colorscheme': 'cyanCarbon'}})
        elif style == 'sphere':
            viewer.setStyle({'sphere': {'scale': 0.25, 'colorscheme': 'cyanCarbon'}})
        elif style == 'line':
            viewer.setStyle({'line': {'linewidth': 1.5, 'colorscheme': 'cyanCarbon'}})
        elif style == 'cartoon':
            viewer.setStyle({'cartoon': {'colorscheme': 'cyanCarbon'}})
        
        # Add surface if requested
        if surface:
            viewer.addSurface(py3Dmol.VDW, {'opacity': 0.6, 'color': 'white'})
        
        # Set view options
        viewer.setBackgroundColor(bgcolor)
        viewer.zoomTo()
        viewer.zoom(zoom)
        
        # Enable spin if requested
        if spin:
            viewer.spin(True)
        
        # Generate HTML
        html_content = viewer.render()
        return html_content
    
    @staticmethod
    def display_single_molecule(smiles=None, mol=None, pdb=None, 
                              title="Molecule Viewer", 
                              enable_download=True,
                              **kwargs):
        """
        Display a single molecule in the Streamlit app
        
        Args:
            smiles: SMILES string of the molecule
            mol: RDKit molecule object (alternative to SMILES)
            pdb: PDB string (alternative to SMILES or mol)
            title: Title to display above the viewer
            enable_download: Whether to add a download button
            **kwargs: Additional arguments for _mol_to_html
            
        Returns:
            None (displays in Streamlit)
        """
        st.subheader(title)
        
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                st.info(f"SMILES: {smiles}")
            except Exception as e:
                st.error(f"Error parsing SMILES: {e}")
                return
        
        if pdb and not mol:
            try:
                viewer = py3Dmol.view(**kwargs)
                viewer.addModel(pdb, 'pdb')
                viewer.setStyle({'cartoon': {'color': 'spectrum'}})
                viewer.zoomTo()
                html_content = viewer.render()
                html(html_content, height=kwargs.get('size', (400, 300))[1])
                return
            except Exception as e:
                st.error(f"Error parsing PDB: {e}")
                return
        
        if mol:
            try:
                # Create control panel
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    style = st.selectbox("Style", 
                                      ["stick", "sphere", "line", "cartoon"], 
                                      key=f"style_{title}")
                    
                    surface = st.checkbox("Show Surface", 
                                       key=f"surface_{title}")
                    
                    spin = st.checkbox("Spin", 
                                    key=f"spin_{title}")
                    
                    bgcolor = st.color_picker("Background", 
                                           "#FFFFFF", 
                                           key=f"bg_{title}")
                    
                    zoom = st.slider("Zoom", 
                                  min_value=0.1, 
                                  max_value=2.0, 
                                  value=1.0, 
                                  key=f"zoom_{title}")
                
                with col1:
                    html_content = MoleculeViewer._mol_to_html(
                        mol, 
                        style=style, 
                        surface=surface, 
                        spin=spin, 
                        bgcolor=bgcolor, 
                        zoom=zoom,
                        **kwargs
                    )
                    
                    html(html_content, height=kwargs.get('size', (400, 300))[1])
                    
                    if enable_download:
                        # Create download buttons
                        mol_block = Chem.MolToMolBlock(mol)
                        smiles_str = Chem.MolToSmiles(mol)
                        
                        col1a, col1b, col1c = st.columns(3)
                        with col1a:
                            MoleculeViewer._download_button(
                                mol_block,
                                "Download MOL",
                                f"{title.replace(' ', '_')}.mol"
                            )
                        with col1b:
                            MoleculeViewer._download_button(
                                smiles_str,
                                "Download SMILES",
                                f"{title.replace(' ', '_')}.smi"
                            )
                        with col1c:
                            # Generate 2D image
                            img = MoleculeViewer._mol_to_image(mol)
                            if img:
                                st.download_button(
                                    "Download PNG",
                                    img,
                                    file_name=f"{title.replace(' ', '_')}.png",
                                    mime="image/png"
                                )
                
            except Exception as e:
                st.error(f"Error rendering molecule: {e}")
        else:
            st.warning("No valid molecule provided")
    
    @staticmethod
    def display_multiple_molecules(molecules, col_width=1, **kwargs):
        """
        Display multiple molecules in a grid layout
        
        Args:
            molecules: List of dicts with molecule information
                       Each dict should have either 'smiles', 'mol', or 'pdb'
                       Can also have 'title', 'properties', etc.
            col_width: Width of each column (1 = full width, 2 = half width, etc.)
            **kwargs: Additional arguments for _mol_to_html
            
        Returns:
            None (displays in Streamlit)
        """
        if not molecules:
            st.warning("No molecules to display")
            return
        
        # Create grid layout
        cols = st.columns(col_width)
        
        for i, mol_data in enumerate(molecules):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                title = mol_data.get('title', f"Molecule {i+1}")
                
                # Display properties if available
                properties = mol_data.get('properties', {})
                if properties:
                    with st.expander(f"{title} Properties"):
                        for prop_name, prop_value in properties.items():
                            st.write(f"{prop_name}: {prop_value}")
                
                # Display molecule
                MoleculeViewer.display_single_molecule(
                    smiles=mol_data.get('smiles'),
                    mol=mol_data.get('mol'),
                    pdb=mol_data.get('pdb'),
                    title=title,
                    enable_download=mol_data.get('enable_download', True),
                    **kwargs
                )
    
    @staticmethod
    def display_molecule_comparison(molecules, titles, properties=None, **kwargs):
        """
        Display molecules side by side for comparison
        
        Args:
            molecules: List of molecules (SMILES strings or RDKit mol objects)
            titles: List of titles for each molecule
            properties: List of dicts containing properties for comparison
            **kwargs: Additional arguments for _mol_to_html
            
        Returns:
            None (displays in Streamlit)
        """
        if not molecules:
            st.warning("No molecules to compare")
            return
        
        num_mols = len(molecules)
        cols = st.columns(num_mols)
        
        # Prepare molecules
        prepared_mols = []
        for i, molecule in enumerate(molecules):
            if isinstance(molecule, str):  # SMILES
                try:
                    mol = Chem.MolFromSmiles(molecule)
                    prepared_mols.append(mol)
                except Exception as e:
                    st.error(f"Error parsing SMILES {i+1}: {e}")
                    prepared_mols.append(None)
            else:  # Assume RDKit mol
                prepared_mols.append(molecule)
        
        # Display molecules side by side
        for i, (mol, col) in enumerate(zip(prepared_mols, cols)):
            with col:
                title = titles[i] if i < len(titles) else f"Molecule {i+1}"
                st.subheader(title)
                
                if mol:
                    html_content = MoleculeViewer._mol_to_html(mol, **kwargs)
                    html(html_content, height=kwargs.get('size', (400, 300))[1])
                    
                    # Display properties if available
                    if properties and i < len(properties):
                        with st.expander("Properties"):
                            for name, value in properties[i].items():
                                st.write(f"{name}: {value}")
                else:
                    st.warning("Invalid molecule")
    
    @staticmethod
    def _mol_to_image(mol, size=(300, 200), highlight_atoms=None):
        """Generate a PNG image of the molecule"""
        if mol is None:
            return None
            
        try:
            from rdkit.Chem import Draw
            import io
            
            if highlight_atoms:
                img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
            else:
                img = Draw.MolToImage(mol, size=size)
                
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None
    
    @staticmethod
    def _download_button(data, label, filename):
        """Create a download button for the given data"""
        if isinstance(data, str):
            data = data.encode()
            
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def similarity_search(query_mol, database_mols, n_results=5, **kwargs):
        """
        Perform similarity search and display results
        
        Args:
            query_mol: Query molecule (SMILES string or RDKit mol)
            database_mols: List of database molecules (SMILES strings or RDKit mols)
            n_results: Number of results to display
            **kwargs: Additional arguments for display_multiple_molecules
            
        Returns:
            List of similarity scores and molecules
        """
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        # Convert query to RDKit mol if needed
        if isinstance(query_mol, str):
            query_mol = Chem.MolFromSmiles(query_mol)
        
        if query_mol is None:
            st.error("Invalid query molecule")
            return []
        
        # Generate query fingerprint
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        
        # Process database molecules
        db_mols = []
        db_fps = []
        
        for i, mol in enumerate(database_mols):
            # Convert to RDKit mol if needed
            if isinstance(mol, str):
                mol_obj = Chem.MolFromSmiles(mol)
                if mol_obj:
                    db_mols.append(mol_obj)
                    db_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol_obj, 2, nBits=2048))
            else:
                db_mols.append(mol)
                db_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        
        # Calculate similarities
        similarities = []
        for i, fp in enumerate(db_fps):
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            similarities.append((sim, db_mols[i], i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Display results
        st.subheader(f"Top {min(n_results, len(similarities))} Similar Molecules")
        
        # Display query molecule
        st.write("Query Molecule:")
        MoleculeViewer.display_single_molecule(
            mol=query_mol, 
            title="Query", 
            **kwargs
        )
        
        # Display results
        result_mols = []
        for i, (sim, mol, idx) in enumerate(similarities[:n_results]):
            result_mols.append({
                'mol': mol,
                'title': f"Rank {i+1} (Similarity: {sim:.3f})",
                'properties': {'Similarity': f"{sim:.3f}", 'Original Index': idx}
            })
        
        MoleculeViewer.display_multiple_molecules(
            result_mols, 
            col_width=min(3, n_results),
            **kwargs
        )
        
        return similarities[:n_results]
    
    @staticmethod
    def substructure_search(query_smarts, database_mols, n_results=5, **kwargs):
        """
        Perform substructure search and display results
        
        Args:
            query_smarts: Query substructure (SMARTS string)
            database_mols: List of database molecules (SMILES strings or RDKit mols)
            n_results: Number of results to display
            **kwargs: Additional arguments for display_multiple_molecules
            
        Returns:
            List of matching molecules with atom indices
        """
        # Check query
        try:
            query_mol = Chem.MolFromSmarts(query_smarts)
            if not query_mol:
                raise ValueError("Invalid SMARTS pattern")
        except Exception as e:
            st.error(f"Error in SMARTS pattern: {e}")
            return []
        
        # Find matches
        matches = []
        
        for i, mol in enumerate(database_mols):
            # Convert to RDKit mol if needed
            if isinstance(mol, str):
                mol_obj = Chem.MolFromSmiles(mol)
            else:
                mol_obj = mol
                
            if mol_obj:
                match_atoms = mol_obj.GetSubstructMatches(query_mol)
                if match_atoms:
                    matches.append((mol_obj, match_atoms, i))
        
        # Display results
        st.subheader(f"Found {len(matches)} Matching Molecules")
        
        # Display query
        st.write(f"Query SMARTS: `{query_smarts}`")
        
        # Display results
        result_mols = []
        for i, (mol, atom_matches, idx) in enumerate(matches[:n_results]):
            # Create molecule with highlighted substructure
            highlighted_mol = Chem.Mol(mol)
            
            # Flatten the list of atom matches
            highlight_atoms = [atom for match in atom_matches for atom in match]
            
            result_mols.append({
                'mol': mol,
                'title': f"Match {i+1}",
                'properties': {
                    'Matching Atoms': str(highlight_atoms),
                    'Original Index': idx,
                    'Number of Matches': len(atom_matches)
                }
            })
            
        MoleculeViewer.display_multiple_molecules(
            result_mols, 
            col_width=min(3, n_results),
            **kwargs
        )
        
        return matches[:n_results]