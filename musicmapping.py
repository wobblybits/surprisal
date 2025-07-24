import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Tuple, Dict
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicGenerator:
    """
    A class to generate musical sequences using various music generation models.
    Supports both DancingIguana/music-generation and sander-wood/tunesformer models.
    """
    
    def __init__(self, model_name: str = "DancingIguana/music-generation", device: str = None):
        """
        Initialize the music generator with a music generation model.
        
        Args:
            model_name: Name of the music generation model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading music generation model {model_name} on device {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Music generation model loaded successfully")
    
    def is_tunesformer_model(self) -> bool:
        """Check if the current model is TunesFormer (uses ABC notation)."""
        return "tunesformer" in self.model_name.lower()
    
    def generate_abc_control_codes(self, num_sections: int = 2, bars_per_section: int = 8, 
                                  edit_distance: int = 5) -> str:
        """
        Generate ABC control codes for TunesFormer model.
        
        Args:
            num_sections: Number of sections (S:1-8)
            bars_per_section: Number of bars per section (B:1-32)
            edit_distance: Edit distance similarity (E:0-10)
            
        Returns:
            ABC control code string
        """
        # Clamp values to valid ranges
        num_sections = max(1, min(8, num_sections))
        bars_per_section = max(1, min(32, bars_per_section))
        edit_distance = max(0, min(10, edit_distance))
        
        abc_header = f"""S:{num_sections}
B:{bars_per_section}
E:{edit_distance}
B:{bars_per_section}
L:1/8
M:3/4
K:D
 de |"D" """
        
        return abc_header
    
    def calculate_surprisal_for_token(self, logit_value: float) -> float:
        """
        Calculate surprisal using the formula: ln(1 + e^-z) where z is the logit value.
        
        Args:
            logit_value: The logit value from the model
            
        Returns:
            Surprisal value
        """
        return np.log(1 + np.exp(-logit_value))
    
    def find_best_musical_token(self, context: str, target_surprisal: float, 
                               top_k: int = 50) -> Tuple[str, float, float]:
        """
        Find the best musical token based on surprisal similarity.
        
        Args:
            context: Current musical context
            target_surprisal: Target surprisal value to match
            top_k: Number of top tokens to consider
            
        Returns:
            Tuple of (chosen_token, actual_surprisal, surprisal_difference)
        """
        # Encode the context
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for next token position
        
        # Get top-k tokens
        top_logits, top_indices = torch.topk(logits, top_k)
        
        best_token = None
        best_surprisal = None
        min_difference = float('inf')
        
        # Evaluate each top token
        for logit_value, token_id in zip(top_logits, top_indices):
            # Calculate surprisal for this token
            surprisal = self.calculate_surprisal_for_token(logit_value.item())
            
            # Calculate difference from target surprisal
            difference = abs(surprisal - target_surprisal)
            
            if difference < min_difference:
                min_difference = difference
                best_token = self.tokenizer.decode([token_id.item()])
                best_surprisal = surprisal
        
        return best_token, best_surprisal, min_difference
    
    def generate_music_from_surprisal(self, surprisal_values: List[float], 
                                     max_notes: int = None, 
                                     start_context: str = None,
                                     use_normalization: bool = False,
                                     source_model: str = None,
                                     target_model: str = None,
                                     logit_analyzer: LogitAnalyzer = None,
                                     abc_control_codes: str = None) -> Dict:
        """
        Generate a musical sequence based on surprisal values using the music generation model.
        
        Args:
            surprisal_values: List of surprisal values to match
            max_notes: Maximum number of notes to generate
            start_context: Starting musical context (default based on model type)
            use_normalization: Whether to normalize surprisal values between models
            source_model: Name of the source model (for normalization)
            target_model: Name of the target model (for normalization)
            logit_analyzer: LogitAnalyzer instance for normalization
            abc_control_codes: ABC control codes for TunesFormer model
            
        Returns:
            Dictionary containing the generated music sequence and metadata
        """
        if not surprisal_values:
            if self.is_tunesformer_model():
                start_context = abc_control_codes or self.generate_abc_control_codes()
            else:
                start_context = start_context or "nsC3s0p25"
            
            return {
                'music_sequence': start_context,
                'surprisal_mapping': {},
                'total_tokens': 1,
                'model_name': self.model_name,
                'notation_type': 'ABC' if self.is_tunesformer_model() else 'Custom'
            }
        
        # Filter out None values and get valid surprisals
        valid_surprisals = [s for s in surprisal_values if s is not None]
        if not valid_surprisals:
            if self.is_tunesformer_model():
                start_context = abc_control_codes or self.generate_abc_control_codes()
            else:
                start_context = start_context or "nsC3s0p25"
            
            return {
                'music_sequence': start_context,
                'surprisal_mapping': {},
                'total_tokens': 1,
                'model_name': self.model_name,
                'notation_type': 'ABC' if self.is_tunesformer_model() else 'Custom'
            }
        
        # Apply normalization if requested
        if use_normalization and logit_analyzer and source_model and target_model:
            logger.info(f"Normalizing surprisal values from {source_model} to {target_model}")
            normalized_surprisals = logit_analyzer.normalize_surprisal_values(
                valid_surprisals, source_model, target_model
            )
            # Filter out None values again after normalization
            normalized_surprisals = [s for s in normalized_surprisals if s is not None]
            if normalized_surprisals:
                valid_surprisals = normalized_surprisals
                logger.info(f"Applied normalization: {len(valid_surprisals)} valid surprisals")
        
        # Normalize surprisal values to 0-1 range
        min_surprisal = min(valid_surprisals)
        max_surprisal = max(valid_surprisals)
        if max_surprisal == min_surprisal:
            normalized_surprisals = [0.5] * len(valid_surprisals)
        else:
            normalized_surprisals = [(s - min_surprisal) / (max_surprisal - min_surprisal) for s in valid_surprisals]
        
        # Set appropriate start context based on model type
        if self.is_tunesformer_model():
            start_context = abc_control_codes or self.generate_abc_control_codes()
        else:
            start_context = start_context or "nsC3s0p25"
        
        # Generate music sequence
        current_context = start_context
        generated_sequence = start_context
        surprisal_mapping = {}
        
        max_notes = max_notes or len(normalized_surprisals)
        notes_to_generate = min(max_notes, len(normalized_surprisals))
        
        logger.info(f"Generating {notes_to_generate} musical tokens based on surprisal values")
        
        for i in range(notes_to_generate):
            target_surprisal = normalized_surprisals[i]
            
            # Find the best musical token for this surprisal value
            chosen_token, actual_surprisal, surprisal_difference = self.find_best_musical_token(
                current_context, target_surprisal
            )
            
            # Add the chosen token to the sequence
            generated_sequence += chosen_token
            current_context += chosen_token
            
            surprisal_mapping[i] = {
                'target_surprisal': target_surprisal,
                'chosen_token': chosen_token,
                'actual_surprisal': actual_surprisal,
                'surprisal_difference': surprisal_difference
            }
            
            logger.info(f"Token {i+1}: {chosen_token} (target: {target_surprisal:.3f}, actual: {actual_surprisal:.3f})")
        
        return {
            'music_sequence': generated_sequence,
            'surprisal_mapping': surprisal_mapping,
            'total_tokens': len(surprisal_mapping) + 1,
            'model_name': self.model_name,
            'original_surprisals': surprisal_values,
            'normalized_surprisals': normalized_surprisals,
            'used_normalization': use_normalization,
            'notation_type': 'ABC' if self.is_tunesformer_model() else 'Custom'
        }
    
    def generate_music_from_text(self, text: str, calculator: 'SurprisalCalculator', 
                                max_notes: int = None,
                                use_normalization: bool = False,
                                logit_analyzer: LogitAnalyzer = None,
                                abc_control_codes: str = None) -> Dict:
        """
        Generate music directly from text by first calculating surprisal values.
        
        Args:
            text: Input text to analyze
            calculator: SurprisalCalculator instance
            max_notes: Maximum number of notes to generate
            use_normalization: Whether to normalize surprisal values between models
            logit_analyzer: LogitAnalyzer instance for normalization
            abc_control_codes: ABC control codes for TunesFormer model
            
        Returns:
            Dictionary containing the generated music sequence and metadata
        """
        # Calculate surprisal values from text
        surprisal_results = calculator.calculate_surprisal(text)
        surprisal_values = [r['surprisal'] for r in surprisal_results]
        
        # Generate music from surprisal values with optional normalization
        return self.generate_music_from_surprisal(
            surprisal_values, 
            max_notes, 
            use_normalization=use_normalization,
            source_model=calculator.model_name,
            target_model=self.model_name,
            logit_analyzer=logit_analyzer,
            abc_control_codes=abc_control_codes
        )
    
    def decode_music_sequence(self, music_sequence: str) -> Dict:
        """
        Decode a music sequence into human-readable format.
        
        Supports two formats:
        1. Custom notation (DancingIguana/music-generation):
           - Notes: ns[pitch]s[duration] (e.g., nsC4s0p25, nsF7s1p0)
           - Rests: rs[duration] (e.g., rs0p5, rs1q6)
           - Chords: cs[count]s[pitches separated by "s"]s[duration] (e.g., cs2sE7sF7s1q3)
           
        2. ABC notation (TunesFormer):
           - Standard ABC notation format
           - Control codes: S:2, B:8, E:5, etc.
           - Musical elements: notes, rests, chords in ABC format
        
        Special character replacements (for custom notation):
        - . = p
        - / = q
        - # = w
        - * = t
        
        Args:
            music_sequence: The generated music sequence
            
        Returns:
            Dictionary with decoded music information
        """
        # Check if this is ABC notation (TunesFormer model)
        if self.is_tunesformer_model() or music_sequence.startswith(('S:', 'X:', 'L:', 'M:', 'K:')):
            return self._decode_abc_notation(music_sequence)
        else:
            return self._decode_custom_notation(music_sequence)
    
    def _decode_abc_notation(self, music_sequence: str) -> Dict:
        """
        Decode ABC notation from TunesFormer model.
        
        Args:
            music_sequence: ABC notation string
            
        Returns:
            Dictionary with decoded ABC music information
        """
        lines = music_sequence.split('\n')
        decoded_elements = []
        control_codes = {}
        musical_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse control codes
            if line.startswith('S:') or line.startswith('B:') or line.startswith('E:'):
                parts = line.split(':')
                if len(parts) == 2:
                    control_codes[parts[0]] = parts[1]
                decoded_elements.append({
                    'type': 'control_code',
                    'code': line,
                    'position': len(decoded_elements)
                })
            elif line.startswith('L:') or line.startswith('M:') or line.startswith('K:'):
                # ABC header information
                decoded_elements.append({
                    'type': 'abc_header',
                    'header': line,
                    'position': len(decoded_elements)
                })
            elif line.startswith('X:'):
                # Tune number
                decoded_elements.append({
                    'type': 'tune_number',
                    'number': line,
                    'position': len(decoded_elements)
                })
            else:
                # Musical content
                musical_content.append(line)
                decoded_elements.append({
                    'type': 'musical_content',
                    'content': line,
                    'position': len(decoded_elements)
                })
        
        return {
            'decoded_elements': decoded_elements,
            'raw_sequence': music_sequence,
            'total_elements': len(decoded_elements),
            'control_codes': control_codes,
            'musical_content': musical_content,
            'notation_type': 'ABC'
        }
    
    def _decode_custom_notation(self, music_sequence: str) -> Dict:
        """
        Decode custom notation from DancingIguana/music-generation model.
        
        Args:
            music_sequence: Custom notation string
            
        Returns:
            Dictionary with decoded custom music information
        """
        decoded_elements = []
        current_pos = 0
        element_count = 0
        
        while current_pos < len(music_sequence):
            element_count += 1
            
            if music_sequence[current_pos:current_pos+2] == "ns":
                # Note: ns[pitch]s[duration]
                current_pos += 2
                pitch_end = music_sequence.find("s", current_pos)
                if pitch_end != -1:
                    pitch = music_sequence[current_pos:pitch_end]
                    current_pos = pitch_end + 1
                    duration_end = music_sequence.find("s", current_pos)
                    if duration_end != -1:
                        duration = music_sequence[current_pos:duration_end]
                        # Decode special characters in duration
                        duration = self._decode_special_chars(duration)
                        decoded_elements.append({
                            'type': 'note',
                            'pitch': pitch,
                            'duration': duration,
                            'position': element_count
                        })
                        current_pos = duration_end + 1
                    else:
                        # Handle case where duration might be at the end
                        duration = music_sequence[current_pos:]
                        duration = self._decode_special_chars(duration)
                        decoded_elements.append({
                            'type': 'note',
                            'pitch': pitch,
                            'duration': duration,
                            'position': element_count
                        })
                        break
                else:
                    break
                    
            elif music_sequence[current_pos:current_pos+2] == "rs":
                # Rest: rs[duration]
                current_pos += 2
                duration_end = music_sequence.find("s", current_pos)
                if duration_end != -1:
                    duration = music_sequence[current_pos:duration_end]
                    duration = self._decode_special_chars(duration)
                    decoded_elements.append({
                        'type': 'rest',
                        'duration': duration,
                        'position': element_count
                    })
                    current_pos = duration_end + 1
                else:
                    # Handle case where duration might be at the end
                    duration = music_sequence[current_pos:]
                    duration = self._decode_special_chars(duration)
                    decoded_elements.append({
                        'type': 'rest',
                        'duration': duration,
                        'position': element_count
                    })
                    break
                    
            elif music_sequence[current_pos:current_pos+2] == "cs":
                # Chord: cs[count]s[pitches separated by "s"]s[duration]
                current_pos += 2
                count_end = music_sequence.find("s", current_pos)
                if count_end != -1:
                    try:
                        count = int(music_sequence[current_pos:count_end])
                        current_pos = count_end + 1
                        
                        # Extract pitches
                        pitches = []
                        for _ in range(count):
                            pitch_end = music_sequence.find("s", current_pos)
                            if pitch_end != -1:
                                pitch = music_sequence[current_pos:pitch_end]
                                pitches.append(pitch)
                                current_pos = pitch_end + 1
                            else:
                                break
                        
                        # Get duration
                        duration_end = music_sequence.find("s", current_pos)
                        if duration_end != -1:
                            duration = music_sequence[current_pos:duration_end]
                            duration = self._decode_special_chars(duration)
                            decoded_elements.append({
                                'type': 'chord',
                                'count': count,
                                'pitches': pitches,
                                'duration': duration,
                                'position': element_count
                            })
                            current_pos = duration_end + 1
                        else:
                            # Handle case where duration might be at the end
                            duration = music_sequence[current_pos:]
                            duration = self._decode_special_chars(duration)
                            decoded_elements.append({
                                'type': 'chord',
                                'count': count,
                                'pitches': pitches,
                                'duration': duration,
                                'position': element_count
                            })
                            break
                    except ValueError:
                        # If count is not a valid integer, skip this element
                        current_pos += 1
                else:
                    break
            else:
                # Skip unknown characters
                current_pos += 1
        
        return {
            'decoded_elements': decoded_elements,
            'raw_sequence': music_sequence,
            'total_elements': len(decoded_elements),
            'element_types': self._count_element_types(decoded_elements),
            'notation_type': 'Custom'
        }
    
    def _decode_special_chars(self, text: str) -> str:
        """
        Decode special character replacements in duration strings.
        
        Args:
            text: Text containing special characters
            
        Returns:
            Decoded text with special characters replaced
        """
        replacements = {
            '.': 'p',
            '/': 'q', 
            '#': 'w',
            '*': 't'
        }
        
        decoded = text
        for old_char, new_char in replacements.items():
            decoded = decoded.replace(old_char, new_char)
        
        return decoded
    
    def _count_element_types(self, elements: List[Dict]) -> Dict[str, int]:
        """
        Count the number of each type of musical element.
        
        Args:
            elements: List of decoded musical elements
            
        Returns:
            Dictionary with counts for each element type
        """
        counts = {'notes': 0, 'rests': 0, 'chords': 0}
        for element in elements:
            if element['type'] == 'note':
                counts['notes'] += 1
            elif element['type'] == 'rest':
                counts['rests'] += 1
            elif element['type'] == 'chord':
                counts['chords'] += 1
        return counts
    
    def format_decoded_music(self, decoded_result: Dict) -> str:
        """
        Format the decoded music into a human-readable string.
        
        Args:
            decoded_result: Result from decode_music_sequence
            
        Returns:
            Formatted string representation of the music
        """
        formatted_lines = []
        formatted_lines.append(f"Music Sequence Analysis")
        formatted_lines.append(f"=" * 50)
        formatted_lines.append(f"Raw sequence: {decoded_result['raw_sequence']}")
        formatted_lines.append(f"Total elements: {decoded_result['total_elements']}")
        formatted_lines.append(f"Element types: {decoded_result['element_types']}")
        formatted_lines.append("")
        formatted_lines.append("Decoded elements:")
        
        for element in decoded_result['decoded_elements']:
            if element['type'] == 'note':
                formatted_lines.append(f"  {element['position']}. Note: {element['pitch']} (Duration: {element['duration']})")
            elif element['type'] == 'rest':
                formatted_lines.append(f"  {element['position']}. Rest (Duration: {element['duration']})")
            elif element['type'] == 'chord':
                pitches_str = ", ".join(element['pitches'])
                formatted_lines.append(f"  {element['position']}. Chord: [{pitches_str}] (Duration: {element['duration']})")
        
        return "\n".join(formatted_lines)

    def convert_to_tonejs_format(self, decoded_result: Dict, bpm: float = 120) -> Dict:
        """
        Convert decoded music to Tone.js-compatible JSON format.
        
        Tone.js expects a format like:
        {
            "bpm": 120,
            "tracks": [
                {
                    "notes": [
                        {"time": "0:0:0", "note": "C4", "duration": "4n"},
                        {"time": "0:1:0", "note": "D4", "duration": "4n"}
                    ]
                }
            ]
        }
        
        Args:
            decoded_result: Result from decode_music_sequence
            bpm: Beats per minute for the sequence
            
        Returns:
            Tone.js-compatible JSON object
        """
        import json
        
        # Convert duration format to Tone.js format
        def convert_duration(duration_str: str) -> str:
            """Convert duration string to Tone.js format."""
            # Map common duration patterns to Tone.js notation
            duration_mapping = {
                '0p25': '16n',  # Sixteenth note
                '0p5': '8n',    # Eighth note
                '1p0': '4n',    # Quarter note
                '1p5': '4n.',   # Dotted quarter note
                '2p0': '2n',    # Half note
                '4p0': '1n',    # Whole note
                '1q6': '4n',    # Quarter note (alternative notation)
                '0p75': '8n.',  # Dotted eighth note
                '0p125': '32n', # Thirty-second note
            }
            
            # Try exact match first
            if duration_str in duration_mapping:
                return duration_mapping[duration_str]
            
            # Try to parse numeric duration
            try:
                # Handle decimal notation (e.g., 0.25, 0.5)
                if 'p' in duration_str:
                    parts = duration_str.split('p')
                    if len(parts) == 2:
                        whole = float(parts[0])
                        decimal = float('0.' + parts[1])
                        total = whole + decimal
                        
                        # Map to closest Tone.js duration
                        if total <= 0.125:
                            return '32n'
                        elif total <= 0.25:
                            return '16n'
                        elif total <= 0.5:
                            return '8n'
                        elif total <= 1.0:
                            return '4n'
                        elif total <= 2.0:
                            return '2n'
                        else:
                            return '1n'
            except:
                pass
            
            # Default to quarter note if unknown
            return '4n'
        
        # Convert pitch format to standard notation
        def convert_pitch(pitch_str: str) -> str:
            """Convert pitch string to standard notation."""
            # Handle special characters in pitch
            pitch_str = pitch_str.replace('w', '#')  # w = sharp
            return pitch_str
        
        # Build Tone.js sequence
        current_time = 0
        notes = []
        
        for element in decoded_result['decoded_elements']:
            duration = convert_duration(element['duration'])
            
            if element['type'] == 'note':
                note_obj = {
                    "time": f"0:{current_time}:0",
                    "note": convert_pitch(element['pitch']),
                    "duration": duration
                }
                notes.append(note_obj)
                
            elif element['type'] == 'chord':
                # Handle chords by creating multiple notes at the same time
                for pitch in element['pitches']:
                    note_obj = {
                        "time": f"0:{current_time}:0",
                        "note": convert_pitch(pitch),
                        "duration": duration
                    }
                    notes.append(note_obj)
            
            # For rests, we just advance time without adding notes
            # Tone.js handles rests automatically when there are gaps
            
            # Advance time (simplified - in practice you'd calculate based on duration)
            current_time += 1
        
        # Create Tone.js format
        tonejs_format = {
            "bpm": bpm,
            "tracks": [
                {
                    "name": "Generated Music",
                    "notes": notes
                }
            ],
            "metadata": {
                "total_elements": decoded_result['total_elements'],
                "element_types": decoded_result['element_types'],
                "raw_sequence": decoded_result['raw_sequence']
            }
        }
        
        return tonejs_format
    
    def export_tonejs_json(self, decoded_result: Dict, bpm: float = 120, filename: str = None) -> str:
        """
        Export decoded music as Tone.js JSON file.
        
        Args:
            decoded_result: Result from decode_music_sequence
            bpm: Beats per minute for the sequence
            filename: Optional filename to save the JSON file
            
        Returns:
            JSON string representation
        """
        import json
        
        tonejs_data = self.convert_to_tonejs_format(decoded_result, bpm)
        json_string = json.dumps(tonejs_data, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_string)
            print(f"Tone.js JSON saved to {filename}")
        
        return json_string

class SurprisalCalculator:
    """
    A class to calculate surprisal values for tokens in text using transformer models.
    Surprisal is calculated as ln(1 + e^-z) where z is the logit value.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
        Initialize the surprisal calculator with a transformer model.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on device {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def analyze_logit_distribution(self, sample_texts: List[str] = None, num_samples: int = 100) -> Dict:
        """
        Analyze the logit distribution of this model.
        
        Args:
            sample_texts: Optional list of sample texts to analyze
            num_samples: Number of samples to generate if no texts provided
            
        Returns:
            Dictionary with logit statistics
        """
        analyzer = LogitAnalyzer()
        return analyzer.analyze_model_logits(
            self.model_name, self.tokenizer, self.model, sample_texts, num_samples
        )
    
    def calculate_surprisal(self, text: str) -> List[Dict]:
        """
        Calculate surprisal values for each token in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing token info and surprisal values
        """
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        logger.info(f"Tokenized text into {len(tokens)} tokens")
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(token_ids)
            logits = outputs.logits
            
        # Calculate surprisal for each token
        surprisal_results = []
        
        for i in range(len(tokens)):
            if i == 0:
                # For the first token, we can't calculate surprisal as there's no context
                surprisal_results.append({
                    'token': tokens[i],
                    'token_id': token_ids[0][i].item(),
                    'surprisal': None,
                    'logit': None,
                    'position': i
                })
                continue
            
            # Get the logit for the current token (predicted by previous tokens)
            # The logit is at position i-1 in the output (since we're predicting token i from context up to i-1)
            current_logits = logits[0, i-1, :]  # Shape: (vocab_size,)
            target_token_id = token_ids[0, i]
            
            # Get the logit value for the actual token
            logit_value = current_logits[target_token_id].item()
            
            # Calculate surprisal using the formula: ln(1 + e^-z)
            surprisal = np.log(1 + np.exp(-logit_value))
            
            surprisal_results.append({
                'token': tokens[i],
                'token_id': target_token_id.item(),
                'surprisal': surprisal,
                'logit': logit_value,
                'position': i
            })
        
        return surprisal_results
    
    def calculate_surprisal_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Calculate surprisal for multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of surprisal results for each text
        """
        results = []
        for text in texts:
            results.append(self.calculate_surprisal(text))
        return results
    
    def get_surprisal_summary(self, surprisal_results: List[Dict]) -> Dict:
        """
        Generate a summary of surprisal statistics.
        
        Args:
            surprisal_results: List of surprisal results from calculate_surprisal
            
        Returns:
            Dictionary with summary statistics
        """
        valid_surprisals = [r['surprisal'] for r in surprisal_results if r['surprisal'] is not None]
        
        if not valid_surprisals:
            return {
                'total_tokens': len(surprisal_results),
                'valid_tokens': 0,
                'mean_surprisal': None,
                'std_surprisal': None,
                'min_surprisal': None,
                'max_surprisal': None
            }
        
        return {
            'total_tokens': len(surprisal_results),
            'valid_tokens': len(valid_surprisals),
            'mean_surprisal': np.mean(valid_surprisals),
            'std_surprisal': np.std(valid_surprisals),
            'min_surprisal': np.min(valid_surprisals),
            'max_surprisal': np.max(valid_surprisals)
        }

class LogitAnalyzer:
    """
    A class to analyze logit distributions across different models
    and provide normalization strategies for surprisal mapping.
    """
    
    def __init__(self):
        """Initialize the logit analyzer."""
        self.model_statistics = {}
    
    def analyze_model_logits(self, model_name: str, tokenizer, model, 
                           sample_texts: List[str] = None, num_samples: int = 100) -> Dict:
        """
        Analyze logit distribution for a given model.
        
        Args:
            model_name: Name of the model
            tokenizer: Model tokenizer
            model: Model instance
            sample_texts: Optional list of sample texts to analyze
            num_samples: Number of random samples to generate if no texts provided
            
        Returns:
            Dictionary with logit statistics
        """
        if sample_texts is None:
            sample_texts = self._generate_sample_texts(num_samples)
        
        all_logits = []
        all_surprisals = []
        
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for text in sample_texts:
                try:
                    # Tokenize and get logits
                    inputs = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
                    inputs = inputs.to(device)
                    
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    # Get logits for each position (excluding the first token)
                    for i in range(1, logits.shape[1]):
                        position_logits = logits[0, i-1, :]  # Logits for predicting token i
                        target_token_id = inputs[0, i]
                        target_logit = position_logits[target_token_id].item()
                        
                        all_logits.append(target_logit)
                        
                        # Calculate surprisal
                        surprisal = np.log(1 + np.exp(-target_logit))
                        all_surprisals.append(surprisal)
                        
                except Exception as e:
                    logger.warning(f"Error processing text '{text[:50]}...': {e}")
                    continue
        
        if not all_logits:
            return {
                'model_name': model_name,
                'error': 'No valid logits found'
            }
        
        # Calculate statistics
        logits_array = np.array(all_logits)
        surprisals_array = np.array(all_surprisals)
        
        stats = {
            'model_name': model_name,
            'logit_stats': {
                'mean': float(np.mean(logits_array)),
                'std': float(np.std(logits_array)),
                'min': float(np.min(logits_array)),
                'max': float(np.max(logits_array)),
                'percentiles': {
                    '5': float(np.percentile(logits_array, 5)),
                    '25': float(np.percentile(logits_array, 25)),
                    '50': float(np.percentile(logits_array, 50)),
                    '75': float(np.percentile(logits_array, 75)),
                    '95': float(np.percentile(logits_array, 95))
                }
            },
            'surprisal_stats': {
                'mean': float(np.mean(surprisals_array)),
                'std': float(np.std(surprisals_array)),
                'min': float(np.min(surprisals_array)),
                'max': float(np.max(surprisals_array)),
                'percentiles': {
                    '5': float(np.percentile(surprisals_array, 5)),
                    '25': float(np.percentile(surprisals_array, 25)),
                    '50': float(np.percentile(surprisals_array, 50)),
                    '75': float(np.percentile(surprisals_array, 75)),
                    '95': float(np.percentile(surprisals_array, 95))
                }
            },
            'sample_count': len(all_logits)
        }
        
        self.model_statistics[model_name] = stats
        return stats
    
    def _generate_sample_texts(self, num_samples: int) -> List[str]:
        """Generate sample texts for logit analysis."""
        sample_texts = [
            "The cat sat on the mat.",
            "Quantum computing is fascinating.",
            "Music brings people together.",
            "The neural network processed the data efficiently.",
            "Artificial intelligence transforms industries.",
            "Machine learning algorithms improve over time.",
            "Deep learning models require large datasets.",
            "Natural language processing enables text understanding.",
            "Computer vision systems recognize patterns.",
            "Reinforcement learning optimizes decision making."
        ]
        
        # Repeat and vary the texts
        texts = []
        for i in range(num_samples):
            base_text = sample_texts[i % len(sample_texts)]
            # Add some variation
            if i > len(sample_texts):
                base_text += f" This is sample {i}."
            texts.append(base_text)
        
        return texts
    
    def compare_models(self, model_names: List[str]) -> Dict:
        """
        Compare logit distributions across multiple models.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Dictionary with comparison statistics
        """
        if len(model_names) < 2:
            return {'error': 'Need at least 2 models to compare'}
        
        comparison = {
            'models': model_names,
            'logit_comparison': {},
            'surprisal_comparison': {},
            'normalization_factors': {}
        }
        
        # Compare logit ranges
        logit_ranges = []
        surprisal_ranges = []
        
        for model_name in model_names:
            if model_name in self.model_statistics:
                stats = self.model_statistics[model_name]
                logit_range = stats['logit_stats']['max'] - stats['logit_stats']['min']
                surprisal_range = stats['surprisal_stats']['max'] - stats['surprisal_stats']['min']
                
                logit_ranges.append(logit_range)
                surprisal_ranges.append(surprisal_range)
                
                comparison['logit_comparison'][model_name] = {
                    'range': logit_range,
                    'mean': stats['logit_stats']['mean'],
                    'std': stats['logit_stats']['std']
                }
                
                comparison['surprisal_comparison'][model_name] = {
                    'range': surprisal_range,
                    'mean': stats['surprisal_stats']['mean'],
                    'std': stats['surprisal_stats']['std']
                }
        
        # Calculate normalization factors
        if logit_ranges:
            max_logit_range = max(logit_ranges)
            for model_name in model_names:
                if model_name in self.model_statistics:
                    current_range = comparison['logit_comparison'][model_name]['range']
                    comparison['normalization_factors'][model_name] = {
                        'logit_scale_factor': max_logit_range / current_range if current_range > 0 else 1.0,
                        'surprisal_scale_factor': max(surprisal_ranges) / comparison['surprisal_comparison'][model_name]['range'] if comparison['surprisal_comparison'][model_name]['range'] > 0 else 1.0
                    }
        
        return comparison
    
    def normalize_surprisal_values(self, surprisal_values: List[float], 
                                 source_model: str, target_model: str) -> List[float]:
        """
        Normalize surprisal values from source model to target model distribution.
        
        Args:
            surprisal_values: List of surprisal values from source model
            source_model: Name of the source model
            target_model: Name of the target model
            
        Returns:
            Normalized surprisal values
        """
        if source_model not in self.model_statistics or target_model not in self.model_statistics:
            logger.warning(f"Model statistics not available for {source_model} or {target_model}")
            return surprisal_values
        
        source_stats = self.model_statistics[source_model]['surprisal_stats']
        target_stats = self.model_statistics[target_model]['surprisal_stats']
        
        # Z-score normalization
        source_mean = source_stats['mean']
        source_std = source_stats['std']
        target_mean = target_stats['mean']
        target_std = target_stats['std']
        
        normalized_values = []
        for surprisal in surprisal_values:
            if surprisal is None:
                normalized_values.append(None)
                continue
            
            # Convert to z-score
            z_score = (surprisal - source_mean) / source_std if source_std > 0 else 0
            
            # Convert to target distribution
            normalized_surprisal = z_score * target_std + target_mean
            
            # Clip to target range
            normalized_surprisal = max(target_stats['min'], min(target_stats['max'], normalized_surprisal))
            normalized_values.append(normalized_surprisal)
        
        return normalized_values
    
    def get_model_statistics(self, model_name: str) -> Dict:
        """Get statistics for a specific model."""
        return self.model_statistics.get(model_name, {})
    
    def print_model_comparison(self, comparison: Dict) -> None:
        """Print a formatted comparison of models."""
        print("Model Logit Distribution Comparison")
        print("=" * 60)
        
        for model_name in comparison['models']:
            if model_name in comparison['logit_comparison']:
                logit_stats = comparison['logit_comparison'][model_name]
                surprisal_stats = comparison['surprisal_comparison'][model_name]
                
                print(f"\n{model_name}:")
                print(f"  Logit range: {logit_stats['range']:.4f}")
                print(f"  Logit mean: {logit_stats['mean']:.4f}")
                print(f"  Logit std: {logit_stats['std']:.4f}")
                print(f"  Surprisal range: {surprisal_stats['range']:.4f}")
                print(f"  Surprisal mean: {surprisal_stats['mean']:.4f}")
                print(f"  Surprisal std: {surprisal_stats['std']:.4f}")
        
        if 'normalization_factors' in comparison:
            print(f"\nNormalization Factors:")
            for model_name, factors in comparison['normalization_factors'].items():
                print(f"  {model_name}:")
                print(f"    Logit scale factor: {factors['logit_scale_factor']:.4f}")
                print(f"    Surprisal scale factor: {factors['surprisal_scale_factor']:.4f}")

def main():
    """Example usage of the SurprisalCalculator and MusicGenerator."""
    
    # Initialize the calculator and music generator
    calculator = SurprisalCalculator(model_name="gpt2")
    music_gen = MusicGenerator()
    
    # Example texts
    texts = [
        "The cat sat on the mat. The cat was black. The cat looked out the window. It saw another cat outside. The cat outside was white.",
        "Quantum computing is fascinating. It dreams up new soft drinks. It is angry in a totally new country.",
        "Music brings people together. It is a new way to express yourself. It is a new way to connect with others. It is actually very old."
    ]
    
    print("=== Surprisal Analysis and Music Generation ===\n")
    
    for i, text in enumerate(texts, 1):
        print(f"Text {i}: {text}")
        print("-" * 50)
        
        # Calculate surprisal
        results = calculator.calculate_surprisal(text)
        
        # Print token-by-token results
        for result in results:
            if result['surprisal'] is not None:
                print(f"Token: '{result['token']}' | Surprisal: {result['surprisal']:.4f} | Logit: {result['logit']:.4f}")
            else:
                print(f"Token: '{result['token']}' | Surprisal: N/A (first token)")
        
        # Print summary
        summary = calculator.get_surprisal_summary(results)
        print(f"\nSummary:")
        print(f"  Mean surprisal: {summary['mean_surprisal']:.4f}")
        print(f"  Std surprisal: {summary['std_surprisal']:.4f}")
        print(f"  Min surprisal: {summary['min_surprisal']:.4f}")
        print(f"  Max surprisal: {summary['max_surprisal']:.4f}")
        
        # Generate music from surprisal values
        surprisal_values = [r['surprisal'] for r in results]
        music_result = music_gen.generate_music_from_surprisal(surprisal_values, max_notes=10)
        
        print(f"\nGenerated Music (Model: {music_result['model_name']}):")
        print(f"  Music Sequence: {music_result['music_sequence']}")
        print(f"  Total tokens: {music_result['total_tokens']}")
        
        # Show surprisal mapping for first few tokens
        print(f"\nSurprisal Mapping (first 5 tokens):")
        mapping_count = min(5, len(music_result['surprisal_mapping']))
        for i in range(mapping_count):
            mapping = music_result['surprisal_mapping'][i]
            print(f"  Token {i+1}: {mapping['chosen_token']} (target: {mapping['target_surprisal']:.3f}, actual: {mapping['actual_surprisal']:.3f})")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
