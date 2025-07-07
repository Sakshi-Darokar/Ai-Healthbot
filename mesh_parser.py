# mesh_parser.py
import xml.etree.ElementTree as ET

def load_mesh_synonyms(xml_path="desc2025.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mesh_dict = {}

    for descriptor in root.findall("DescriptorRecord"):
        try:
            main_heading = descriptor.find("DescriptorName/String").text.strip()
            for concept in descriptor.findall("ConceptList/Concept"):
                for term in concept.findall("TermList/Term"):
                    synonym = term.find("String").text.strip().lower()
                    mesh_dict[synonym] = main_heading
        except Exception as e:
            print("Error parsing:", e)

    return mesh_dict
