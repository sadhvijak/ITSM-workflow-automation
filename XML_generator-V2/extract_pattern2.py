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
    - Concrete XML examples (full depth, not truncated)
    - Attribute value enums
    - Element co-occurrence rules
    - Context-specific patterns
    - RAG-optimized natural language descriptions
    """

    def __init__(self):
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

        self.child_ordering_patterns = defaultdict(lambda: defaultdict(list))
        self.xml_examples = defaultdict(list)
        self.attribute_enums = defaultdict(lambda: {
            "values": Counter(),
            "pattern": None,
            "is_enum": False,
            "is_reference": False,
            "is_dynamic": False
        })
        self.cooccurrence_rules = defaultdict(lambda: {
            "always_with": set(),
            "never_with": set(),
            "usually_with": defaultdict(int),
            "conditional_on": defaultdict(list)
        })
        self.context_patterns = defaultdict(lambda: defaultdict(dict))
        self.element_contexts = defaultdict(list)
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
        self.element_siblings_tracker = defaultdict(lambda: defaultdict(int))

        # === NEW: Store full XML snippets per element (no depth truncation) ===
        self.full_xml_examples = defaultdict(list)

    # ──────────────────────────────────────────────────────────────────────────
    # EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────

    def extract_from_files(self, xml_files: List[str]):
        print(f"\n{'='*70}")
        print("ENHANCED PATTERN EXTRACTION (with ordering, examples, enums)")
        print(f"{'='*70}")
        print(f"Processing {len(xml_files)} files...\n")

        for idx, xml_file in enumerate(xml_files, 1):
            print(f"[{idx}/{len(xml_files)}] {Path(xml_file).name}")
            try:
                self._process_file(xml_file)
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\n{'='*70}")
        print("Pattern extraction complete!")
        print(f"{'='*70}\n")

    def _process_file(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        root_tag = self._clean_tag(root.tag)
        self.root_patterns[root_tag] += 1

        if root.tag.startswith('{'):
            ns = root.tag[1:root.tag.index('}')]
            self.namespace_info[root_tag] = ns

        self._analyze_element(root, parent_tag=None, depth=0, siblings=[], parent_elem=None)
        self._analyze_cooccurrence_rules()

    def _analyze_element(self, elem, parent_tag=None, depth=0, siblings=[], parent_elem=None):
        tag = self._clean_tag(elem.tag)

        if not tag or tag.startswith('#'):
            return

        pattern = self.element_patterns[tag]
        pattern["occurrences"] += 1
        pattern["depth_range"]["min"] = min(pattern["depth_range"]["min"], depth)
        pattern["depth_range"]["max"] = max(pattern["depth_range"]["max"], depth)

        if parent_tag:
            pattern["possible_parents"].add(parent_tag)
            self.parent_child_patterns[parent_tag][tag] += 1

        # === IMPROVED: Store full XML snippet (deeper depth = more useful for RAG) ===
        xml_snippet_short = self._element_to_string(elem, max_depth=2)
        xml_snippet_full  = self._element_to_string(elem, max_depth=6)  # Full depth

        if len(self.xml_examples[tag]) < 5:
            self.xml_examples[tag].append({
                "xml": xml_snippet_short,
                "parent": parent_tag,
                "depth": depth,
                "file_context": "example"
            })

        # === NEW: Store full-depth examples separately for RAG chunks ===
        if len(self.full_xml_examples[tag]) < 3:
            self.full_xml_examples[tag].append(xml_snippet_full)

        context_key = f"{parent_tag or 'root'}"
        context = pattern["contexts"][context_key]
        context["parent"] = parent_tag
        if len(context["examples"]) < 3:
            context["examples"].append(xml_snippet_short)

        for attr_name, attr_val in elem.attrib.items():
            if pattern["occurrences"] == 1:
                pattern["required_attributes"].add(attr_name)
            elif attr_name not in pattern["required_attributes"]:
                pattern["optional_attributes"].add(attr_name)
                if attr_name in pattern["required_attributes"]:
                    pattern["required_attributes"].remove(attr_name)
                    pattern["optional_attributes"].add(attr_name)

            self._analyze_attribute(attr_name, attr_val, tag)
            self._track_attribute_enum(attr_name, attr_val, tag)

        if elem.text and elem.text.strip():
            pattern["can_have_text"] = True
            text_type = self._infer_data_type(elem.text.strip())
            pattern["text_data_types"].add(text_type)

            if len(list(elem)) == 0:
                self._track_attribute_enum(f"{tag}_text_value", elem.text.strip(), tag)

        children = [c for c in elem if not self._clean_tag(c.tag).startswith('#')]
        child_tags = [self._clean_tag(c.tag) for c in children]

        if child_tags:
            self.child_ordering_patterns[tag][context_key].append(child_tags)
            context["child_sequences"].append(child_tags)

            for child_tag in child_tags:
                pattern["required_children"][child_tag] += 1

        if parent_tag:
            self.element_contexts[tag].append({
                "parent": parent_tag,
                "siblings": [s for s in siblings if s != tag],
                "children": child_tags,
                "depth": depth,
                "xml_snippet": xml_snippet_short[:500]
            })

        for sibling in siblings:
            if sibling != tag:
                self.element_siblings_tracker[tag][sibling] += 1

        if parent_tag and siblings:
            position = siblings.index(tag) if tag in siblings else -1
            if position == 0:
                self.position_rules[tag]["must_be_first"] += 1
            elif position == len(siblings) - 1:
                self.position_rules[tag]["must_be_last"] += 1
            else:
                self.position_rules[tag]["can_be_anywhere"] += 1
            self.position_rules[tag]["typical_positions"].append(position)

        for i, sibling_tag in enumerate(child_tags):
            if i > 0:
                prev_sibling = child_tags[i-1]
                self.sibling_patterns[prev_sibling][sibling_tag] += 1

        if children and len(child_tags) > 1:
            self.sequence_patterns.append((tag, child_tags))

        self._extract_reference_patterns(elem, tag)

        for child in children:
            self._analyze_element(child, tag, depth + 1, child_tags, elem)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _track_attribute_enum(self, attr_name: str, attr_value: str, element_tag: str):
        enum_pattern = self.attribute_enums[attr_name]
        enum_pattern["values"][attr_value] += 1

        unique_values = len(enum_pattern["values"])
        total_occurrences = sum(enum_pattern["values"].values())

        if total_occurrences >= 3:
            if unique_values <= 10:
                enum_pattern["is_enum"] = True
            elif unique_values / total_occurrences < 0.3:
                enum_pattern["is_enum"] = True

        if any(ref in attr_name.lower() for ref in ['ref', 'reference', 'target', 'connector']):
            enum_pattern["is_reference"] = True

        if '{!' in attr_value or '$' in attr_value:
            enum_pattern["is_dynamic"] = True

    def _element_to_string(self, elem, max_depth=6, current_depth=0) -> str:
        """
        Convert element to indented XML string.
        max_depth=6 by default so full element structures are captured.
        """
        if current_depth >= max_depth:
            tag = self._clean_tag(elem.tag)
            return f"<{tag}>...</{tag}>"

        tag = self._clean_tag(elem.tag)
        attrs = ' '.join([f'{k}="{v}"' for k, v in elem.attrib.items()])
        attrs_str = f" {attrs}" if attrs else ""
        indent = "    " * current_depth

        if len(list(elem)) == 0:
            text = elem.text.strip() if elem.text else ""
            if text:
                return f"{indent}<{tag}{attrs_str}>{text}</{tag}>"
            else:
                return f"{indent}<{tag}{attrs_str}/>"

        children_str = ""
        for child in elem:
            child_tag = self._clean_tag(child.tag)
            if not child_tag.startswith('#'):
                children_str += "\n" + self._element_to_string(child, max_depth, current_depth + 1)

        return f"{indent}<{tag}{attrs_str}>{children_str}\n{indent}</{tag}>"

    def _analyze_cooccurrence_rules(self):
        for element, siblings in self.element_siblings_tracker.items():
            total_occurrences = self.element_patterns[element]["occurrences"]

            for sibling, cooccur_count in siblings.items():
                cooccurrence_ratio = cooccur_count / total_occurrences

                if cooccurrence_ratio >= 0.95:
                    self.cooccurrence_rules[element]["always_with"].add(sibling)
                elif cooccurrence_ratio >= 0.5:
                    self.cooccurrence_rules[element]["usually_with"][sibling] = cooccur_count

        all_elements = set(self.element_patterns.keys())
        for element in all_elements:
            appearing_siblings = set(self.element_siblings_tracker[element].keys())
            possible_siblings = set()

            for parent in self.element_patterns[element]["possible_parents"]:
                possible_siblings.update(self.parent_child_patterns[parent].keys())

            possible_siblings.discard(element)

            never_together = possible_siblings - appearing_siblings
            if never_together and len(appearing_siblings) > 0:
                self.cooccurrence_rules[element]["never_with"] = never_together

    def _analyze_attribute(self, attr_name: str, attr_value: str, element_tag: str):
        attr_pattern = self.attribute_patterns[attr_name]

        if any(ref_word in attr_name.lower() for ref_word in ['ref', 'reference', 'target', 'source']):
            attr_pattern["is_reference"] = True

        if any(id_word in attr_name.lower() for id_word in ['id', 'name', 'identifier']):
            attr_pattern["is_identifier"] = True

        data_type = self._infer_data_type(attr_value)
        attr_pattern["data_type"].add(data_type)

    def _extract_reference_patterns(self, elem, tag):
        ref_indicators = ['reference', 'ref', 'target', 'connector', 'id']

        for child in elem:
            child_tag = self._clean_tag(child.tag)

            if any(indicator in child_tag.lower() for indicator in ref_indicators):
                if child.text and child.text.strip():
                    self.reference_patterns[tag]["reference_attributes"].add(child_tag)

    def _infer_data_type(self, value: str) -> str:
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
        return tag.split('}')[-1] if '}' in tag else tag

    def _calculate_canonical_child_order(self, parent: str) -> List[str]:
        if parent not in self.child_ordering_patterns:
            return []

        all_sequences = []
        for context, sequences in self.child_ordering_patterns[parent].items():
            all_sequences.extend(sequences)

        if not all_sequences:
            return []

        sequence_counter = Counter()
        for seq in all_sequences:
            sequence_counter[tuple(seq)] += 1

        if sequence_counter:
            most_common_seq = sequence_counter.most_common(1)[0][0]
            return list(most_common_seq)

        return []

    def _find_common_sequences(self) -> List[Dict]:
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
        result = {}
        for key, value in pattern_dict.items():
            if isinstance(value, dict):
                result[key] = self._serialize_patterns(value)
            elif isinstance(value, set):
                result[key] = sorted(list(value))
            elif isinstance(value, defaultdict):
                result[key] = self._serialize_patterns(dict(value))
            elif isinstance(value, Counter):
                result[key] = dict(value)
            else:
                result[key] = value
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # PATTERN GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def generate_generic_patterns(self) -> Dict:
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
            pattern["canonical_child_order"] = self._calculate_canonical_child_order(tag)

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

        sequence_templates = self._find_common_sequences()

        processed_enums = {}
        for attr_name, enum_data in self.attribute_enums.items():
            if enum_data["is_enum"]:
                processed_enums[attr_name] = {
                    "possible_values": [val for val, _ in enum_data["values"].most_common()],
                    "most_common": enum_data["values"].most_common(5),
                    "is_reference": enum_data["is_reference"],
                    "is_dynamic": enum_data["is_dynamic"]
                }

        patterns = {
            "metadata": {
                "pattern_type": "enhanced_xml_structure",
                "purpose": "RAG-based XML generation with ordering, examples, and enums",
                "version": "3.0"
            },
            "root_element_patterns": {
                "possible_roots": list(self.root_patterns.keys()),
                "most_common_root": self.root_patterns.most_common(1)[0][0] if self.root_patterns else None
            },
            "element_patterns": self._serialize_patterns(self.element_patterns),
            "child_ordering_rules": {
                parent: {
                    "canonical_order": self._calculate_canonical_child_order(parent),
                    "all_observed_orders": [
                        list(seq)
                        for context, seqs in self.child_ordering_patterns[parent].items()
                        for seq in seqs[:3]
                    ]
                }
                for parent in self.child_ordering_patterns.keys()
            },
            "xml_examples": self._serialize_patterns(self.xml_examples),
            "full_xml_examples": {k: v for k, v in self.full_xml_examples.items()},
            "attribute_enums": processed_enums,
            "cooccurrence_rules": self._serialize_patterns(self.cooccurrence_rules),
            "context_patterns": self._serialize_patterns(
                {elem: dict(contexts)
                 for elem, contexts in self.element_patterns.items()
                 if contexts.get("contexts")}
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

    # ──────────────────────────────────────────────────────────────────────────
    # RAG CHUNK DESCRIPTIONS  ← KEY IMPROVEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_element_description(self, elem_name: str, pattern: Dict, canonical_order: List[str]) -> str:
        """
        Generate a RICH natural language description so ChromaDB semantic search
        can find this chunk when the agent asks about specific elements.

        OLD: "Element 'recordLookups'; MUST contain: object..."
        NEW: "recordLookups XML element structure for Salesforce Flow.
              Used in AutoLaunched and Record-Triggered flows to query Salesforce records.
              MUST contain: name, label, locationX, locationY, object, filters, filterLogic.
              Child order: name -> label -> locationX -> locationY -> filterLogic -> filters -> object
              Example: <recordLookups>..."
        """
        parts = []

        # Element name — repeated for embedding weight
        parts.append(f"XML element '{elem_name}' structure and usage in Salesforce Flow metadata.")

        # What it's for (inferred from name)
        usage_map = {
            "recordLookups":  "Used to query/retrieve Salesforce records (SOQL GET). Used in AutoLaunched and Record-Triggered flows.",
            "recordUpdates":  "Used to update existing Salesforce records. Used in AutoLaunched and Record-Triggered flows.",
            "recordCreates":  "Used to create new Salesforce records. Used in AutoLaunched and Record-Triggered flows.",
            "recordDeletes":  "Used to delete Salesforce records.",
            "decisions":      "Used to branch flow logic based on conditions. Contains rules with conditionLogic.",
            "assignments":    "Used to assign values to variables. Contains assignmentItems.",
            "screens":        "Used in Screen Flows to display UI to users. Contains fields (NOT screenFields).",
            "loops":          "Used to iterate over a collection of records.",
            "subflows":       "Used to call another flow from within this flow.",
            "variables":      "Declares input/output/local variables used throughout the flow.",
            "formulas":       "Declares formula expressions used in the flow.",
            "start":          "The entry point of the flow. Required in all flows. Contains connector to first element.",
            "connector":      "Links one flow element to the next. Contains targetReference.",
            "filters":        "Defines filter criteria for record queries. Used inside recordLookups and start.",
            "rules":          "Defines a branch outcome inside a decisions element. Contains conditionLogic and conditions.",
            "conditions":     "Individual condition inside a rules element. Contains leftValueReference, operator, rightValue.",
            "inputAssignments": "Maps a value to a field when creating or updating records.",
            "outputAssignments": "Maps a queried field value to a variable.",
            "fields":         "Screen field inside a screens element. Use fieldType DisplayText or InputField.",
            "scheduledPaths": "Defines a scheduled path inside a record-triggered flow start element.",
        }

        if elem_name in usage_map:
            parts.append(usage_map[elem_name])

        # Required children
        if pattern.get("required_children"):
            parts.append(f"MUST contain these child elements: {', '.join(pattern['required_children'])}.")

        # Optional children
        if pattern.get("optional_children"):
            parts.append(f"MAY also contain: {', '.join(list(pattern['optional_children'])[:8])}.")

        # Canonical child order — CRITICAL for XML validity
        if canonical_order:
            parts.append(
                f"CORRECT child element order (CRITICAL): {' -> '.join(canonical_order)}. "
                f"Elements placed in wrong order will cause deployment errors."
            )

        # Parent context
        if pattern.get("possible_parents"):
            parts.append(f"Appears inside: {', '.join(list(pattern['possible_parents'])[:5])}.")

        # Text content
        if pattern.get("can_have_text"):
            types = list(pattern.get("text_data_types", []))
            parts.append(f"Contains text content of type: {', '.join(types)}.")

        # Full XML example
        full_examples = self.full_xml_examples.get(elem_name, [])
        if full_examples:
            parts.append(f"XML EXAMPLE:\n{full_examples[0]}")

        return "\n".join(parts)

    def _generate_child_ordering_description(self, parent: str, canonical_order: List[str], all_orders: List) -> str:
        """Rich description for child ordering RAG chunks."""
        parts = [
            f"Child element ordering rules for '{parent}' in Salesforce Flow XML.",
            f"When writing a '{parent}' element, its children MUST appear in this order:",
            f"  {' -> '.join(canonical_order)}",
            f"Placing children in the wrong order causes Salesforce deployment errors.",
        ]

        if all_orders:
            unique = list({tuple(o) for o in all_orders[:5]})
            parts.append(f"Observed orderings from real flows: {[list(o) for o in unique[:3]]}")

        return "\n".join(parts)

    def _generate_cooccurrence_description(self, elem: str, rules: Dict) -> str:
        """Rich description for co-occurrence RAG chunks."""
        parts = [f"Co-occurrence rules for element '{elem}' in Salesforce Flow XML."]

        if rules.get("always_with"):
            parts.append(
                f"'{elem}' ALWAYS appears together with: {', '.join(list(rules['always_with'])[:5])}. "
                f"If you include '{elem}', you MUST also include these elements."
            )

        if rules.get("never_with"):
            never = list(rules['never_with'])[:5]
            if never:
                parts.append(
                    f"'{elem}' NEVER appears together with: {', '.join(never)}. "
                    f"These elements are mutually exclusive with '{elem}'."
                )

        if rules.get("usually_with"):
            usually = list(rules["usually_with"].keys())[:5]
            parts.append(f"'{elem}' usually appears alongside: {', '.join(usually)}.")

        return "\n".join(parts)

    def _generate_relationship_description(self, parent: str, children: Dict) -> str:
        """Rich description for parent-child relationship RAG chunks."""
        child_list = list(children.keys())
        freq_info = ", ".join([f"{c}({children[c]}x)" for c in child_list[:8]])

        return (
            f"Parent-child relationship: '{parent}' element in Salesforce Flow XML.\n"
            f"'{parent}' can contain these child elements: {', '.join(child_list[:10])}.\n"
            f"Frequency of each child: {freq_info}.\n"
            f"Use this when building a '{parent}' element to know what children are allowed."
        )

    def _generate_sequence_description(self, seq_template: Dict) -> str:
        """Rich description for sequence template RAG chunks."""
        parent = seq_template["parent"]
        seq = seq_template["child_sequence"]
        freq = seq_template["frequency"]

        return (
            f"Common element sequence pattern in Salesforce Flow XML.\n"
            f"Inside '{parent}', this child sequence appears {freq} times across real flows:\n"
            f"  {' -> '.join(seq)}\n"
            f"Use this as a template when constructing '{parent}' elements."
        )

    def _generate_positional_description(self, elem: str, constraint: Dict) -> str:
        """Rich description for positional constraint RAG chunks."""
        return (
            f"Positional rule for '{elem}' element in Salesforce Flow XML.\n"
            f"Constraint: {constraint['constraint']} (confidence: {constraint['confidence']:.0%}).\n"
            f"When placing '{elem}' inside its parent, follow this positioning rule."
        )

    def _generate_enum_description(self, attr_name: str, enum_info: Dict) -> str:
        """Rich description for attribute enum RAG chunks."""
        values = enum_info["possible_values"][:15]
        most_common = [v for v, _ in enum_info.get("most_common", [])[:5]]

        return (
            f"Valid values for '{attr_name}' in Salesforce Flow XML.\n"
            f"This is an enum field — only these values are accepted: {', '.join(str(v) for v in values)}.\n"
            f"Most commonly used values: {', '.join(str(v) for v in most_common)}.\n"
            f"Using any other value will cause a deployment error.\n"
            f"Note: Values are case-sensitive (PascalCase where applicable)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────────────────

    def export_for_rag(self, output_file: str):
        """Export in RAG-friendly format with RICH descriptions for semantic search."""
        patterns = self.generate_generic_patterns()

        rag_chunks = []

        # ── Chunk Type 1: Element templates with full examples ─────────────────
        for elem_name, elem_pattern in patterns["element_patterns"].items():
            examples = patterns["xml_examples"].get(elem_name, [])
            example_xml = examples[0]["xml"] if examples else "No example available"
            canonical_order = elem_pattern.get("canonical_child_order", [])

            chunk = {
                "type": "element_template",
                "element": elem_name,
                "pattern": elem_pattern,
                "xml_example": example_xml,
                # Full depth XML example for richer context
                "full_xml_example": patterns["full_xml_examples"].get(elem_name, [""])[0],
                "canonical_child_order": canonical_order,
                # RICH description — what ChromaDB actually searches against
                "description": self._generate_element_description(elem_name, elem_pattern, canonical_order)
            }
            rag_chunks.append(chunk)

        # ── Chunk Type 2: Child ordering rules ────────────────────────────────
        for parent, order_info in patterns["child_ordering_rules"].items():
            canonical = order_info["canonical_order"]
            if not canonical:
                continue

            chunk = {
                "type": "child_ordering_rule",
                "parent": parent,
                "canonical_order": canonical,
                "all_observed_orders": order_info.get("all_observed_orders", []),
                "description": self._generate_child_ordering_description(
                    parent, canonical, order_info.get("all_observed_orders", [])
                )
            }
            rag_chunks.append(chunk)

        # ── Chunk Type 3: Attribute enums ─────────────────────────────────────
        for attr_name, enum_info in patterns["attribute_enums"].items():
            chunk = {
                "type": "attribute_enum",
                "attribute": attr_name,
                "possible_values": enum_info["possible_values"],
                "most_common": enum_info["most_common"],
                "description": self._generate_enum_description(attr_name, enum_info)
            }
            rag_chunks.append(chunk)

        # ── Chunk Type 4: Co-occurrence rules ─────────────────────────────────
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

        # ── Chunk Type 5: Parent-child relationships ──────────────────────────
        for parent, children in patterns["relationship_patterns"]["parent_child"].items():
            chunk = {
                "type": "parent_child_relationship",
                "parent": parent,
                "children": children,
                "description": self._generate_relationship_description(parent, children)
            }
            rag_chunks.append(chunk)

        # ── Chunk Type 6: Sequence templates ─────────────────────────────────
        for seq_template in patterns["constraint_patterns"]["sequence_templates"]:
            chunk = {
                "type": "sequence_template",
                "pattern": seq_template,
                "description": self._generate_sequence_description(seq_template)
            }
            rag_chunks.append(chunk)

        # ── Chunk Type 7: Positional constraints ──────────────────────────────
        for elem, constraint in patterns["constraint_patterns"]["positional"].items():
            chunk = {
                "type": "positional_constraint",
                "element": elem,
                "constraint": constraint,
                "description": self._generate_positional_description(elem, constraint)
            }
            rag_chunks.append(chunk)

        output = {
            "full_patterns": patterns,
            "rag_chunks": rag_chunks,
            "summary": {
                "total_elements": len(patterns["element_patterns"]),
                "total_chunks": len(rag_chunks),
                "chunk_types": {
                    "element_templates":    len([c for c in rag_chunks if c["type"] == "element_template"]),
                    "ordering_rules":       len([c for c in rag_chunks if c["type"] == "child_ordering_rule"]),
                    "attribute_enums":      len([c for c in rag_chunks if c["type"] == "attribute_enum"]),
                    "cooccurrence_rules":   len([c for c in rag_chunks if c["type"] == "cooccurrence_rule"]),
                    "relationships":        len([c for c in rag_chunks if c["type"] == "parent_child_relationship"]),
                    "sequences":            len([c for c in rag_chunks if c["type"] == "sequence_template"]),
                    "positional":           len([c for c in rag_chunks if c["type"] == "positional_constraint"])
                }
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ ENHANCED RAG-ready patterns exported to: {output_file}")
        print(f"\nSummary:")
        print(f"  - {len(patterns['element_patterns'])} element patterns")
        print(f"  - {len(patterns.get('child_ordering_rules', {}))} ordering rules")
        print(f"  - {len(patterns.get('attribute_enums', {}))} attribute enums")
        print(f"  - {len(patterns.get('full_xml_examples', {}))} elements with full XML examples")
        print(f"  - {len(rag_chunks)} total RAG chunks")
        print(f"\nChunk breakdown:")
        for chunk_type, count in output["summary"]["chunk_types"].items():
            print(f"  - {chunk_type}: {count}")

        return output_file

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY PRINT
    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self):
        print(f"\n{'='*70}")
        print("ENHANCED EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"Element types discovered: {len(self.element_patterns)}")
        print(f"Parent-child patterns: {len(self.parent_child_patterns)}")
        print(f"Child ordering patterns: {len(self.child_ordering_patterns)}")
        print(f"Sequence patterns: {len(self.sequence_patterns)}")
        print(f"Attribute types: {len(self.attribute_patterns)}")
        print(f"Attribute enums detected: {sum(1 for e in self.attribute_enums.values() if e['is_enum'])}")
        print(f"Full XML examples collected: {sum(len(ex) for ex in self.full_xml_examples.values())}")
        print(f"Co-occurrence rules: {len(self.cooccurrence_rules)}")

        print(f"\nTop elements by frequency:")
        sorted_elements = sorted(
            self.element_patterns.items(),
            key=lambda x: x[1]["occurrences"],
            reverse=True
        )[:10]

        for elem, pattern in sorted_elements:
            canonical_order = self._calculate_canonical_child_order(elem)
            order_str = f" [Order: {' -> '.join(canonical_order[:3])}...]" if canonical_order else ""
            print(f"  - {elem}: {pattern['occurrences']} occurrences{order_str}")

        print(f"\nDetected attribute enums (top 10):")
        enum_attrs = [(name, data) for name, data in self.attribute_enums.items() if data["is_enum"]]
        enum_attrs.sort(key=lambda x: sum(x[1]["values"].values()), reverse=True)

        for attr_name, enum_data in enum_attrs[:10]:
            values = [v for v, _ in enum_data["values"].most_common(5)]
            print(f"  - {attr_name}: {values}")

        print(f"\nSample co-occurrence rules:")
        sample_rules = list(self.cooccurrence_rules.items())[:5]
        for elem, rules in sample_rules:
            if rules.get("always_with"):
                print(f"  - '{elem}' always appears with: {list(rules['always_with'])[:3]}")
            if rules.get("never_with"):
                never = list(rules['never_with'])[:3]
                if never:
                    print(f"  - '{elem}' never appears with: {never}")

        print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
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

    valid_files = [f for f in xml_files if Path(f).exists()]

    if not valid_files:
        print("No valid XML files found!")
        print("Please check that the XML files exist in the 'scripts/' directory")
        return

    print(f"\nFound {len(valid_files)} valid XML files")

    extractor = EnhancedXMLPatternExtractor()
    extractor.extract_from_files(valid_files)
    extractor.print_summary()

    output_file = "enhanced_xml_patterns_with recordtrigger_for_rag2.json"
    extractor.export_for_rag(output_file)

    print(f"\n{'='*70}")
    print("READY FOR RAG EMBEDDING!")
    print(f"{'='*70}")
    print(f"\nOutput file: {output_file}")
    print(f"\nWhat's improved in this version:")
    print(f"  - Full-depth XML examples (not truncated at depth 2)")
    print(f"  - Rich natural language descriptions per chunk")
    print(f"  - Element usage context (what each element is FOR)")
    print(f"  - Descriptions mention element names multiple times for better embedding recall")
    print(f"\nNext steps:")
    print(f"  1. Run embed.py to embed chunks into ChromaDB (use path='./chroma_storage')")
    print(f"  2. Query with specific element names e.g. 'recordLookups structure'")
    print(f"     NOT broad terms like 'Auto-Launched Flow'")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()