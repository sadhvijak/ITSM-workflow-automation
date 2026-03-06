import xml.etree.ElementTree as ET
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


class EnhancedXMLPatternExtractor:
    """
    Extracts COMPREHENSIVE patterns including:
    - Child element ordering (CRITICAL for XML)
    - Concrete XML examples
    - Attribute value enums
    - Element co-occurrence rules
    - Context-specific patterns
    """
    
    def __init__(self):
        # Basic pattern storage
        self.element_patterns = defaultdict(lambda: {
            "occurrences": 0,
            "possible_parents": set(),
            "required_children": defaultdict(int),
            "optional_children": set(),
            "required_attributes": set(),
            "optional_attributes": set(),
            "can_have_text": False,
            "text_data_types": set(),
            "depth_range": {"min": float('inf'), "max": 0},
            "contexts": defaultdict(lambda: {
                "parent": None,
                "examples": [],
                "child_sequences": []
            })
        })
        
        # === NEW: Child Element Ordering ===
        self.child_ordering_patterns = defaultdict(lambda: defaultdict(list))
        # Structure: {parent: {context: [list of child sequences]}}
        
        # === NEW: Concrete XML Examples ===
        self.xml_examples = defaultdict(list)
        # Structure: {element: [list of complete XML snippets]}
        
        # === NEW: Attribute Value Enums ===
        self.attribute_enums = defaultdict(lambda: {
            "values": Counter(),
            "pattern": None,  # regex pattern if applicable
            "is_enum": False,
            "is_reference": False,
            "is_dynamic": False
        })
        # Structure: {attribute_name: {values: Counter, pattern: str}}
        
        # === NEW: Element Co-occurrence Rules ===
        self.cooccurrence_rules = defaultdict(lambda: {
            "always_with": set(),      # Elements that always appear together
            "never_with": set(),        # Mutually exclusive elements
            "usually_with": defaultdict(int),  # Common combinations
            "conditional_on": defaultdict(list)  # If X exists, then Y must/can exist
        })
        
        # === NEW: Context-Specific Patterns ===
        self.context_patterns = defaultdict(lambda: defaultdict(dict))
        # Structure: {element: {parent_context: {specific patterns}}}
        
        # === NEW: Full Element Contexts (captures surrounding structure) ===
        self.element_contexts = defaultdict(list)
        # Structure: {element: [{parent, siblings, children, xml_snippet}]}
        
        # Existing patterns (keeping your original ones)
        self.parent_child_patterns = defaultdict(lambda: defaultdict(int))
        self.sibling_patterns = defaultdict(lambda: defaultdict(int))
        self.reference_patterns = defaultdict(lambda: {
            "can_reference": set(),
            "can_be_referenced_by": set(),
            "reference_attributes": set()
        })
        self.position_rules = defaultdict(lambda: {
            "must_be_first": 0,
            "must_be_last": 0,
            "can_be_anywhere": 0,
            "typical_positions": []
        })
        self.sequence_patterns = []
        self.attribute_patterns = defaultdict(lambda: {
            "data_type": set(),
            "is_reference": False,
            "is_identifier": False,
            "required_on": set(),
            "optional_on": set()
        })
        self.namespace_info = {}
        self.root_patterns = Counter()
        
        # === NEW: Track element siblings for co-occurrence ===
        self.element_siblings_tracker = defaultdict(lambda: defaultdict(int))
        
    def extract_from_files(self, xml_files: List[str]):
        """Extract patterns from multiple XML files"""
        print(f"\n{'='*70}")
        print("ENHANCED PATTERN EXTRACTION (with ordering, examples, enums)")
        print(f"{'='*70}")
        print(f"Processing {len(xml_files)} files...\n")
        
        for idx, xml_file in enumerate(xml_files, 1):
            print(f"[{idx}/{len(xml_files)}] {Path(xml_file).name}")
            try:
                self._process_file(xml_file)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n{'='*70}")
        print("Pattern extraction complete!")
        print(f"{'='*70}\n")
    
    def _process_file(self, xml_file: str):
        """Process a single file to extract patterns"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Track root pattern
        root_tag = self._clean_tag(root.tag)
        self.root_patterns[root_tag] += 1
        
        # Extract namespace pattern
        if root.tag.startswith('{'):
            ns = root.tag[1:root.tag.index('}')]
            self.namespace_info[root_tag] = ns
        
        # Recursively analyze with enhanced tracking
        self._analyze_element(root, parent_tag=None, depth=0, siblings=[], parent_elem=None)
        
        # Extract co-occurrence rules at file level
        self._analyze_cooccurrence_rules()
    
    def _analyze_element(self, elem, parent_tag=None, depth=0, siblings=[], parent_elem=None):
        """Recursively analyze element patterns with ENHANCED tracking"""
        tag = self._clean_tag(elem.tag)
        
        if not tag or tag.startswith('#'):
            return
        
        pattern = self.element_patterns[tag]
        pattern["occurrences"] += 1
        
        # Track depth range
        pattern["depth_range"]["min"] = min(pattern["depth_range"]["min"], depth)
        pattern["depth_range"]["max"] = max(pattern["depth_range"]["max"], depth)
        
        # Track parent relationship
        if parent_tag:
            pattern["possible_parents"].add(parent_tag)
            self.parent_child_patterns[parent_tag][tag] += 1
        
        # === NEW: Extract complete XML snippet as example ===
        xml_snippet = self._element_to_string(elem, max_depth=2)
        if len(self.xml_examples[tag]) < 5:  # Store max 5 examples per element
            self.xml_examples[tag].append({
                "xml": xml_snippet,
                "parent": parent_tag,
                "depth": depth,
                "file_context": "example"
            })
        
        # === NEW: Track context-specific patterns ===
        context_key = f"{parent_tag or 'root'}"
        context = pattern["contexts"][context_key]
        context["parent"] = parent_tag
        if len(context["examples"]) < 3:
            context["examples"].append(xml_snippet)
        
        # Track attributes with ENUM extraction
        for attr_name, attr_val in elem.attrib.items():
            # Basic tracking (your original code)
            if pattern["occurrences"] == 1:
                pattern["required_attributes"].add(attr_name)
            elif attr_name not in pattern["required_attributes"]:
                pattern["optional_attributes"].add(attr_name)
                if attr_name in pattern["required_attributes"]:
                    pattern["required_attributes"].remove(attr_name)
                    pattern["optional_attributes"].add(attr_name)
            
            self._analyze_attribute(attr_name, attr_val, tag)
            
            # === NEW: Track attribute value enums ===
            self._track_attribute_enum(attr_name, attr_val, tag)
        
        # Track text content capability
        if elem.text and elem.text.strip():
            pattern["can_have_text"] = True
            text_type = self._infer_data_type(elem.text.strip())
            pattern["text_data_types"].add(text_type)
            
            # === NEW: Track text value enums for leaf elements ===
            if len(list(elem)) == 0:  # Leaf element
                self._track_attribute_enum(f"{tag}_text_value", elem.text.strip(), tag)
        
        # === CRITICAL NEW: Track child element ordering ===
        children = [c for c in elem if not self._clean_tag(c.tag).startswith('#')]
        child_tags = [self._clean_tag(c.tag) for c in children]
        
        if child_tags:
            # Store the exact sequence of children
            self.child_ordering_patterns[tag][context_key].append(child_tags)
            context["child_sequences"].append(child_tags)
            
            # Track in parent-child patterns
            for child_tag in child_tags:
                pattern["required_children"][child_tag] += 1
        
        # === NEW: Track element context (siblings, parent, children) ===
        if parent_tag:
            self.element_contexts[tag].append({
                "parent": parent_tag,
                "siblings": [s for s in siblings if s != tag],
                "children": child_tags,
                "depth": depth,
                "xml_snippet": xml_snippet[:500]  # Truncate long examples
            })
        
        # === NEW: Track sibling co-occurrence ===
        for sibling in siblings:
            if sibling != tag:
                self.element_siblings_tracker[tag][sibling] += 1
        
        # Track positional patterns (your original code)
        if parent_tag and siblings:
            position = siblings.index(tag) if tag in siblings else -1
            if position == 0:
                self.position_rules[tag]["must_be_first"] += 1
            elif position == len(siblings) - 1:
                self.position_rules[tag]["must_be_last"] += 1
            else:
                self.position_rules[tag]["can_be_anywhere"] += 1
            
            self.position_rules[tag]["typical_positions"].append(position)
        
        # Track sibling patterns (your original code)
        for i, sibling_tag in enumerate(child_tags):
            if i > 0:
                prev_sibling = child_tags[i-1]
                self.sibling_patterns[prev_sibling][sibling_tag] += 1
        
        # Track sequence patterns (your original code)
        if children and len(child_tags) > 1:
            self.sequence_patterns.append((tag, child_tags))
        
        # Track reference patterns
        self._extract_reference_patterns(elem, tag)
        
        # Recurse
        for child in children:
            self._analyze_element(child, tag, depth + 1, child_tags, elem)
    
    def _track_attribute_enum(self, attr_name: str, attr_value: str, element_tag: str):
        """Track possible enum values for attributes"""
        enum_pattern = self.attribute_enums[attr_name]
        
        # Increment value counter
        enum_pattern["values"][attr_value] += 1
        
        # Determine if this looks like an enum (limited set of values)
        unique_values = len(enum_pattern["values"])
        total_occurrences = sum(enum_pattern["values"].values())
        
        # Heuristics for enum detection
        if total_occurrences >= 3:
            if unique_values <= 10:  # Max 10 unique values = likely enum
                enum_pattern["is_enum"] = True
            elif unique_values / total_occurrences < 0.3:  # Low cardinality
                enum_pattern["is_enum"] = True
        
        # Detect reference patterns
        if any(ref in attr_name.lower() for ref in ['ref', 'reference', 'target', 'connector']):
            enum_pattern["is_reference"] = True
        
        # Detect dynamic values (formulas, variables)
        if '{!' in attr_value or '$' in attr_value:
            enum_pattern["is_dynamic"] = True
    
    def _element_to_string(self, elem, max_depth=2, current_depth=0) -> str:
        """Convert element to XML string with limited depth"""
        if current_depth >= max_depth:
            return f"<{self._clean_tag(elem.tag)}>...</{self._clean_tag(elem.tag)}>"
        
        tag = self._clean_tag(elem.tag)
        attrs = ' '.join([f'{k}="{v}"' for k, v in elem.attrib.items()])
        attrs_str = f" {attrs}" if attrs else ""
        
        if len(list(elem)) == 0:  # Leaf element
            text = elem.text.strip() if elem.text else ""
            if text:
                return f"<{tag}{attrs_str}>{text}</{tag}>"
            else:
                return f"<{tag}{attrs_str}/>"
        
        # Has children
        children_str = ""
        for child in elem:
            child_tag = self._clean_tag(child.tag)
            if not child_tag.startswith('#'):
                children_str += "\n  " + self._element_to_string(child, max_depth, current_depth + 1)
        
        return f"<{tag}{attrs_str}>{children_str}\n</{tag}>"
    
    def _analyze_cooccurrence_rules(self):
        """Analyze which elements always/never appear together"""
        # For each element, check its siblings
        for element, siblings in self.element_siblings_tracker.items():
            total_occurrences = self.element_patterns[element]["occurrences"]
            
            for sibling, cooccur_count in siblings.items():
                cooccurrence_ratio = cooccur_count / total_occurrences
                
                if cooccurrence_ratio >= 0.95:  # Appears together 95%+ of the time
                    self.cooccurrence_rules[element]["always_with"].add(sibling)
                elif cooccurrence_ratio >= 0.5:  # Appears together 50%+ of the time
                    self.cooccurrence_rules[element]["usually_with"][sibling] = cooccur_count
        
        # Detect mutually exclusive elements (elements that never appear as siblings)
        all_elements = set(self.element_patterns.keys())
        for element in all_elements:
            appearing_siblings = set(self.element_siblings_tracker[element].keys())
            possible_siblings = set()
            
            # Find elements that share the same parents
            for parent in self.element_patterns[element]["possible_parents"]:
                possible_siblings.update(self.parent_child_patterns[parent].keys())
            
            # Remove self
            possible_siblings.discard(element)
            
            # Elements that could be siblings but never are = mutually exclusive
            never_together = possible_siblings - appearing_siblings
            if never_together and len(appearing_siblings) > 0:
                self.cooccurrence_rules[element]["never_with"] = never_together
    
    def _analyze_attribute(self, attr_name: str, attr_value: str, element_tag: str):
        """Analyze attribute patterns"""
        attr_pattern = self.attribute_patterns[attr_name]
        
        # Determine attribute purpose
        if any(ref_word in attr_name.lower() for ref_word in ['ref', 'reference', 'target', 'source']):
            attr_pattern["is_reference"] = True
        
        if any(id_word in attr_name.lower() for id_word in ['id', 'name', 'identifier']):
            attr_pattern["is_identifier"] = True
        
        # Infer data type from value
        data_type = self._infer_data_type(attr_value)
        attr_pattern["data_type"].add(data_type)
    
    def _extract_reference_patterns(self, elem, tag):
        """Extract which elements can reference which other elements"""
        ref_indicators = ['reference', 'ref', 'target', 'connector', 'id']
        
        for child in elem:
            child_tag = self._clean_tag(child.tag)
            
            # Check if this is a reference element
            if any(indicator in child_tag.lower() for indicator in ref_indicators):
                if child.text and child.text.strip():
                    self.reference_patterns[tag]["reference_attributes"].add(child_tag)
    
    def _infer_data_type(self, value: str) -> str:
        """Infer generic data type from value"""
        value = value.strip()
        
        if value.lower() in ['true', 'false']:
            return "boolean"
        if value.isdigit():
            return "integer"
        try:
            float(value)
            return "number"
        except:
            pass
        if len(value) > 100:
            return "long_text"
        if '{!' in value:
            return "formula_reference"
        return "string"
    
    def _clean_tag(self, tag: str) -> str:
        """Remove namespace from tag"""
        return tag.split('}')[-1] if '}' in tag else tag
    
    def _calculate_canonical_child_order(self, parent: str) -> List[str]:
        """
        Calculate the canonical ordering of children for a parent element
        Returns the most common sequence pattern
        """
        if parent not in self.child_ordering_patterns:
            return []
        
        all_sequences = []
        for context, sequences in self.child_ordering_patterns[parent].items():
            all_sequences.extend(sequences)
        
        if not all_sequences:
            return []
        
        # Find the most common complete sequence
        sequence_counter = Counter()
        for seq in all_sequences:
            sequence_counter[tuple(seq)] += 1
        
        if sequence_counter:
            most_common_seq = sequence_counter.most_common(1)[0][0]
            return list(most_common_seq)
        
        return []
    
    def generate_generic_patterns(self) -> Dict:
        """Generate final generic patterns suitable for RAG"""
        
        # Calculate required vs optional children
        for tag, pattern in self.element_patterns.items():
            total_occurrences = pattern["occurrences"]
            
            required_children = []
            optional_children = []
            
            for child, count in pattern["required_children"].items():
                if count == total_occurrences:
                    required_children.append(child)
                else:
                    optional_children.append(child)
            
            pattern["required_children"] = required_children
            pattern["optional_children"] = optional_children
            
            # === NEW: Add canonical child ordering ===
            pattern["canonical_child_order"] = self._calculate_canonical_child_order(tag)
        
        # Generate positional rules
        position_constraints = {}
        for tag, pos_info in self.position_rules.items():
            total = sum([pos_info["must_be_first"], pos_info["must_be_last"], pos_info["can_be_anywhere"]])
            
            if total == 0:
                continue
            
            constraint = "flexible"
            if pos_info["must_be_first"] == total:
                constraint = "always_first_child"
            elif pos_info["must_be_last"] == total:
                constraint = "always_last_child"
            elif pos_info["must_be_first"] / total > 0.8:
                constraint = "usually_first_child"
            elif pos_info["must_be_last"] / total > 0.8:
                constraint = "usually_last_child"
            
            position_constraints[tag] = {
                "constraint": constraint,
                "confidence": max(pos_info["must_be_first"], pos_info["must_be_last"]) / total if total > 0 else 0
            }
        
        # Find common sequence patterns
        sequence_templates = self._find_common_sequences()
        
        # === NEW: Process attribute enums ===
        processed_enums = {}
        for attr_name, enum_data in self.attribute_enums.items():
            if enum_data["is_enum"]:
                processed_enums[attr_name] = {
                    "possible_values": [val for val, _ in enum_data["values"].most_common()],
                    "most_common": enum_data["values"].most_common(5),
                    "is_reference": enum_data["is_reference"],
                    "is_dynamic": enum_data["is_dynamic"]
                }
        
        # Build final output
        patterns = {
            "metadata": {
                "pattern_type": "enhanced_xml_structure",
                "purpose": "RAG-based XML generation with ordering, examples, and enums",
                "version": "2.0"
            },
            
            "root_element_patterns": {
                "possible_roots": list(self.root_patterns.keys()),
                "most_common_root": self.root_patterns.most_common(1)[0][0] if self.root_patterns else None
            },
            
            "element_patterns": self._serialize_patterns(self.element_patterns),
            
            # === NEW: Child ordering patterns ===
            "child_ordering_rules": {
                parent: {
                    "canonical_order": self._calculate_canonical_child_order(parent),
                    "all_observed_orders": [list(seq) for context, seqs in self.child_ordering_patterns[parent].items() for seq in seqs[:3]]
                }
                for parent in self.child_ordering_patterns.keys()
            },
            
            # === NEW: XML Examples ===
            "xml_examples": self._serialize_patterns(self.xml_examples),
            
            # === NEW: Attribute Enums ===
            "attribute_enums": processed_enums,
            
            # === NEW: Co-occurrence Rules ===
            "cooccurrence_rules": self._serialize_patterns(self.cooccurrence_rules),
            
            # === NEW: Context-Specific Patterns ===
            "context_patterns": self._serialize_patterns(
                {elem: dict(contexts) for elem, contexts in self.element_patterns.items() if contexts.get("contexts")}
            ),
            
            "relationship_patterns": {
                "parent_child": self._serialize_patterns(self.parent_child_patterns),
                "sibling_sequences": self._serialize_patterns(self.sibling_patterns),
                "reference_capabilities": self._serialize_patterns(self.reference_patterns)
            },
            
            "constraint_patterns": {
                "positional": position_constraints,
                "sequence_templates": sequence_templates
            },
            
            "attribute_patterns": self._serialize_patterns(self.attribute_patterns),
            
            "namespace_patterns": self.namespace_info
        }
        
        return patterns
    
    def _find_common_sequences(self) -> List[Dict]:
        """Find common child element sequences"""
        sequence_counter = Counter()
        
        for parent, children in self.sequence_patterns:
            seq = (parent, tuple(children))
            sequence_counter[seq] += 1
        
        common_sequences = []
        for (parent, children), count in sequence_counter.most_common(20):
            if count > 1:
                common_sequences.append({
                    "parent": parent,
                    "child_sequence": list(children),
                    "frequency": count
                })
        
        return common_sequences
    
    def _serialize_patterns(self, pattern_dict):
        """Convert defaultdict with sets to regular dict with lists"""
        result = {}
        for key, value in pattern_dict.items():
            if isinstance(value, dict):
                result[key] = self._serialize_patterns(value)
            elif isinstance(value, set):
                result[key] = sorted(list(value))  # Sort for consistency
            elif isinstance(value, defaultdict):
                result[key] = self._serialize_patterns(dict(value))
            elif isinstance(value, Counter):
                result[key] = dict(value)
            else:
                result[key] = value
        return result
    
    def export_for_rag(self, output_file: str):
        """Export in RAG-friendly format with ENHANCED chunks"""
        patterns = self.generate_generic_patterns()
        
        # Create RAG-optimized chunks
        rag_chunks = []
        
        # Chunk Type 1: Element templates WITH examples and ordering
        for elem_name, elem_pattern in patterns["element_patterns"].items():
            # Get XML examples
            examples = patterns["xml_examples"].get(elem_name, [])
            example_xml = examples[0]["xml"] if examples else "No example available"
            
            # Get canonical order
            canonical_order = elem_pattern.get("canonical_child_order", [])
            
            chunk = {
                "type": "element_template",
                "element": elem_name,
                "pattern": elem_pattern,
                "xml_example": example_xml,
                "canonical_child_order": canonical_order,
                "description": self._generate_element_description(elem_name, elem_pattern, canonical_order)
            }
            rag_chunks.append(chunk)
        
        # Chunk Type 2: Child ordering rules
        for parent, order_info in patterns["child_ordering_rules"].items():
            chunk = {
                "type": "child_ordering_rule",
                "parent": parent,
                "canonical_order": order_info["canonical_order"],
                "description": f"Children of '{parent}' should appear in order: {' -> '.join(order_info['canonical_order'])}"
            }
            rag_chunks.append(chunk)
        
        # Chunk Type 3: Attribute enums (CRITICAL for valid values)
        for attr_name, enum_info in patterns["attribute_enums"].items():
            chunk = {
                "type": "attribute_enum",
                "attribute": attr_name,
                "possible_values": enum_info["possible_values"],
                "most_common": enum_info["most_common"],
                "description": f"Valid values for '{attr_name}': {', '.join(str(v) for v in enum_info['possible_values'][:10])}"
            }
            rag_chunks.append(chunk)
        
        # Chunk Type 4: Co-occurrence rules
        for elem, rules in patterns["cooccurrence_rules"].items():
            if rules.get("always_with") or rules.get("never_with"):
                chunk = {
                    "type": "cooccurrence_rule",
                    "element": elem,
                    "always_with": list(rules.get("always_with", [])),
                    "never_with": list(rules.get("never_with", [])),
                    "description": self._generate_cooccurrence_description(elem, rules)
                }
                rag_chunks.append(chunk)
        
        # Chunk Type 5: Relationship templates
        for parent, children in patterns["relationship_patterns"]["parent_child"].items():
            chunk = {
                "type": "parent_child_relationship",
                "parent": parent,
                "children": children,
                "description": f"{parent} can contain: {', '.join(children.keys())}"
            }
            rag_chunks.append(chunk)
        
        # Chunk Type 6: Sequence templates
        for seq_template in patterns["constraint_patterns"]["sequence_templates"]:
            chunk = {
                "type": "sequence_template",
                "pattern": seq_template,
                "description": f"Common pattern: {seq_template['parent']} contains sequence [{' -> '.join(seq_template['child_sequence'])}]"
            }
            rag_chunks.append(chunk)
        
        # Chunk Type 7: Positional constraints
        for elem, constraint in patterns["constraint_patterns"]["positional"].items():
            chunk = {
                "type": "positional_constraint",
                "element": elem,
                "constraint": constraint,
                "description": f"{elem} positioning rule: {constraint['constraint']}"
            }
            rag_chunks.append(chunk)
        
        output = {
            "full_patterns": patterns,
            "rag_chunks": rag_chunks,
            "summary": {
                "total_elements": len(patterns["element_patterns"]),
                "total_chunks": len(rag_chunks),
                "chunk_types": {
                    "element_templates": len([c for c in rag_chunks if c["type"] == "element_template"]),
                    "ordering_rules": len([c for c in rag_chunks if c["type"] == "child_ordering_rule"]),
                    "attribute_enums": len([c for c in rag_chunks if c["type"] == "attribute_enum"]),
                    "cooccurrence_rules": len([c for c in rag_chunks if c["type"] == "cooccurrence_rule"]),
                    "relationships": len([c for c in rag_chunks if c["type"] == "parent_child_relationship"]),
                    "sequences": len([c for c in rag_chunks if c["type"] == "sequence_template"]),
                    "positional": len([c for c in rag_chunks if c["type"] == "positional_constraint"])
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ ENHANCED RAG-ready patterns exported to: {output_file}")
        print(f"\n📊 Summary:")
        print(f"  - {len(patterns['element_patterns'])} element patterns")
        print(f"  - {len(patterns.get('child_ordering_rules', {}))} ordering rules")
        print(f"  - {len(patterns.get('attribute_enums', {}))} attribute enums")
        print(f"  - {len(patterns.get('xml_examples', {}))} elements with examples")
        print(f"  - {len(rag_chunks)} total RAG chunks")
        print(f"\n📋 Chunk breakdown:")
        for chunk_type, count in output["summary"]["chunk_types"].items():
            print(f"  - {chunk_type}: {count}")
        
        return output_file
    
    def _generate_element_description(self, elem_name: str, pattern: Dict, canonical_order: List[str]) -> str:
        """Generate natural language description of element pattern"""
        parts = [f"Element '{elem_name}'"]
        
        if pattern.get("required_children"):
            parts.append(f"MUST contain: {', '.join(pattern['required_children'])}")
        
        if canonical_order:
            parts.append(f"Child order: {' -> '.join(canonical_order)}")
        
        if pattern.get("optional_children"):
            parts.append(f"MAY contain: {', '.join(pattern['optional_children'][:5])}")
        
        if pattern.get("required_attributes"):
            parts.append(f"Requires attributes: {', '.join(list(pattern['required_attributes'])[:3])}")
        
        if pattern.get("can_have_text"):
            parts.append("Can contain text content")
        
        return "; ".join(parts)
    
    def _generate_cooccurrence_description(self, elem: str, rules: Dict) -> str:
        """Generate description of co-occurrence rules"""
        parts = [f"Element '{elem}'"]
        
        if rules.get("always_with"):
            parts.append(f"ALWAYS appears with: {', '.join(list(rules['always_with'])[:3])}")
        
        if rules.get("never_with"):
            parts.append(f"NEVER appears with: {', '.join(list(rules['never_with'])[:3])}")
        
        return "; ".join(parts)
    
    def print_summary(self):
        """Print extraction summary"""
        print(f"\n{'='*70}")
        print("ENHANCED EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"Element types discovered: {len(self.element_patterns)}")
        print(f"Parent-child patterns: {len(self.parent_child_patterns)}")
        print(f"Child ordering patterns: {len(self.child_ordering_patterns)}")
        print(f"Sequence patterns: {len(self.sequence_patterns)}")
        print(f"Attribute types: {len(self.attribute_patterns)}")
        print(f"Attribute enums detected: {sum(1 for e in self.attribute_enums.values() if e['is_enum'])}")
        print(f"XML examples collected: {sum(len(ex) for ex in self.xml_examples.values())}")
        print(f"Co-occurrence rules: {len(self.cooccurrence_rules)}")
        
        print(f"\n🔝 Top elements by frequency:")
        sorted_elements = sorted(
            self.element_patterns.items(),
            key=lambda x: x[1]["occurrences"],
            reverse=True
        )[:10]
        
        for elem, pattern in sorted_elements:
            canonical_order = self._calculate_canonical_child_order(elem)
            order_str = f" [Order: {' → '.join(canonical_order[:3])}...]" if canonical_order else ""
            print(f"  - {elem}: {pattern['occurrences']} occurrences{order_str}")
        
        print(f"\n📋 Detected attribute enums (top 10):")
        enum_attrs = [(name, data) for name, data in self.attribute_enums.items() if data["is_enum"]]
        enum_attrs.sort(key=lambda x: sum(x[1]["values"].values()), reverse=True)
        
        for attr_name, enum_data in enum_attrs[:10]:
            values = [v for v, _ in enum_data["values"].most_common(5)]
            print(f"  - {attr_name}: {values}")
        
        print(f"\n🔗 Sample co-occurrence rules:")
        sample_rules = list(self.cooccurrence_rules.items())[:5]
        for elem, rules in sample_rules:
            if rules.get("always_with"):
                print(f"  - '{elem}' always appears with: {list(rules['always_with'])[:3]}")
            if rules.get("never_with"):
                never = list(rules['never_with'])[:3]
                if never:
                    print(f"  - '{elem}' never appears with: {never}")
        
        print(f"{'='*70}\n")


# ========================================
# USAGE
# ========================================

def main():
    """Main execution"""
    
    # Your XML files
    xml_files = [
        "scripts/SDO_FSL_Reset_Shift_Dates.flow-meta.xml",
        "scripts/SDO_Service_Create_Change_From_Problem.flow-meta.xml",
        "scripts/SDO_Service_Password_Reset.flow-meta.xml",
        "scripts/Approve_Time_Sheets.flow-meta.xml",
        "scripts/AccountSum_Triggered_flow_PT.flow-meta.xml",
        "scripts/Exchange_RMA_1.flow-meta.xml",
        "scripts/SDO_OMS_NEW_Create_Invoice_and_Ensure_Funds.flow-meta.xml",
        "scripts/Get_Rebate_Type_Payouts_for_Member_1.flow-meta.xml",
        "scripts/SDO_Service_Create_Draft_Article_from_Case.flow-meta.xml",
        "scripts/Process_Change_Billing_Cycle1.flow-meta.xml",
        "Incident_Close_Trigger1.flow-meta.xml",
        "CreateAbsnTimeSheetEntry1.flow-meta.xml",
        "CreateExtlChnlProcExcp1.flow-meta.xml",
        "SendFdbkRdyNtfcn1.flow-meta.xml"
    ]
    
    # Filter existing files
    valid_files = [f for f in xml_files if Path(f).exists()]
    
    if not valid_files:
        print("✗ No valid XML files found!")
        print("Please check that the XML files exist in the 'scripts/' directory")
        return
    
    print(f"\n✓ Found {len(valid_files)} valid XML files")
    
    # Extract ENHANCED patterns
    extractor = EnhancedXMLPatternExtractor()
    extractor.extract_from_files(valid_files)
    
    # Print detailed summary
    extractor.print_summary()
    
    # Export for RAG
    output_file = "enhanced_xml_patterns_with recordtrigger_for_rag.json"
    extractor.export_for_rag(output_file)
    
    print(f"\n{'='*70}")
    print("✅ READY FOR RAG EMBEDDING!")
    print(f"{'='*70}")
    print(f"\n📦 Output file: {output_file}")
    print(f"\n💡 What's included:")
    print(f"  ✓ Child element ordering (CRITICAL for valid XML)")
    print(f"  ✓ Concrete XML examples for each element")
    print(f"  ✓ Attribute value enums (valid values)")
    print(f"  ✓ Element co-occurrence rules (what goes with what)")
    print(f"  ✓ Context-specific patterns")
    print(f"  ✓ All your original patterns plus enhancements")
    print(f"\n🚀 Next steps:")
    print(f"  1. Use the 'rag_chunks' array for vector database embedding")
    print(f"  2. Each chunk type targets specific generation needs:")
    print(f"     - element_template: Complete structure with examples")
    print(f"     - child_ordering_rule: Correct tag sequences")
    print(f"     - attribute_enum: Valid attribute values")
    print(f"     - cooccurrence_rule: What elements go together")
    print(f"     - parent_child_relationship: Nesting rules")
    print(f"     - sequence_template: Common patterns")
    print(f"     - positional_constraint: Placement rules")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()