# OCR and Math Utilities for AI Camera Backend

import re
import base64
import io
import logging
from PIL import Image
from typing import Optional, Dict, List
import sympy as sp
from sympy import symbols, Eq, solve, simplify, expand, factor, latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

logger = logging.getLogger(__name__)

class MathSolver:
    """Advanced mathematical expression and equation solver"""
    
    def __init__(self):
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
    
    def clean_expression(self, expr: str) -> str:
        """Clean and normalize mathematical expressions"""
        # Remove extra whitespace
        expr = re.sub(r'\s+', ' ', expr.strip())
        
        # Common symbol replacements
        replacements = {
            '×': '*', '÷': '/', '−': '-', '±': '+-',
            '²': '**2', '³': '**3', '⁴': '**4',
            'π': 'pi', '∞': 'oo', '√': 'sqrt',
            '∑': 'Sum', '∏': 'Product',
            '∫': 'integrate', '∂': 'diff',
            '≈': '==', '≠': '!=', '≤': '<=', '≥': '>='
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        # Handle implicit multiplication (e.g., 2x -> 2*x)
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)
        expr = re.sub(r'(\))(\()', r'\1*\2', expr)
        
        return expr
    
    def solve_equation(self, expression: str) -> Dict:
        """Solve mathematical equations and expressions"""
        try:
            cleaned_expr = self.clean_expression(expression)
            logger.info(f"Solving: {cleaned_expr}")
            
            steps = []
            result_type = "expression"
            
            # Handle equations (contains =)
            if '=' in cleaned_expr:
                result_type = "equation"
                left, right = cleaned_expr.split('=', 1)
                left_expr = parse_expr(left.strip(), transformations=self.transformations)
                right_expr = parse_expr(right.strip(), transformations=self.transformations)
                
                equation = Eq(left_expr, right_expr)
                steps.append(f"Original equation: {equation}")
                
                # Find variables
                variables = equation.free_symbols
                if variables:
                    x = list(variables)[0]  # Use the first variable
                    solutions = solve(equation, x)
                    
                    if solutions:
                        steps.append(f"Solving for {x}")
                        if len(solutions) == 1:
                            result = str(solutions[0])
                            steps.append(f"Solution: {x} = {result}")
                        else:
                            result = f"{x} = {', '.join(map(str, solutions))}"
                            steps.append(f"Multiple solutions: {result}")
                    else:
                        result = "No solution found"
                        steps.append("No solution exists for this equation")
                else:
                    # No variables, just evaluate equality
                    try:
                        left_val = float(left_expr.evalf())
                        right_val = float(right_expr.evalf())
                        result = "True" if abs(left_val - right_val) < 1e-10 else "False"
                        steps.append(f"Left side: {left_val}")
                        steps.append(f"Right side: {right_val}")
                        steps.append(f"Equation is: {result}")
                    except:
                        result = f"Cannot evaluate: {equation}"
                        steps.append("Cannot numerically evaluate this equation")
            
            else:
                # Handle expressions (no =)
                expr = parse_expr(cleaned_expr, transformations=self.transformations)
                steps.append(f"Original expression: {expr}")
                
                # Try various operations
                try:
                    # Simplify
                    simplified = simplify(expr)
                    if simplified != expr:
                        steps.append(f"Simplified: {simplified}")
                        expr = simplified
                    
                    # Try to factor if polynomial
                    try:
                        factored = factor(expr)
                        if factored != expr:
                            steps.append(f"Factored: {factored}")
                    except:
                        pass
                    
                    # Try to expand if needed
                    try:
                        expanded = expand(expr)
                        if expanded != expr and len(str(expanded)) < len(str(expr)) * 1.5:
                            steps.append(f"Expanded: {expanded}")
                    except:
                        pass
                    
                    # Try to evaluate numerically
                    try:
                        numeric_result = float(expr.evalf())
                        result = str(numeric_result)
                        steps.append(f"Numeric value: {result}")
                    except:
                        result = str(expr)
                        steps.append(f"Final form: {result}")
                        
                except Exception as e:
                    result = cleaned_expr
                    steps.append(f"Could not process: {str(e)}")
            
            return {
                "success": True,
                "solution": result,
                "steps": steps,
                "type": result_type,
                "latex_solution": self._to_latex(result) if result not in ["True", "False", "No solution found"] else None
            }
            
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            return {
                "success": False,
                "solution": f"Error: Could not solve this expression ({str(e)})",
                "steps": [f"Error occurred: {str(e)}"],
                "type": "error",
                "latex_solution": None
            }
    
    def _to_latex(self, expression: str) -> Optional[str]:
        """Convert expression to LaTeX format"""
        try:
            expr = parse_expr(expression, transformations=self.transformations)
            return latex(expr)
        except:
            return None

class OCRProcessor:
    """OCR text processing utilities"""
    
    def __init__(self):
        self.math_patterns = [
            r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Basic arithmetic
            r'\d+\s*[\+\-\*\/]\s*[a-zA-Z]',  # Variable expressions
            r'[a-zA-Z]\s*[\+\-\*\/\=]\s*\d+',  # Variable equations
            r'\d+[a-zA-Z]',  # Coefficient notation
            r'[a-zA-Z]\^?\d+',  # Powers
            r'√\d+',  # Square roots
            r'\d+²',  # Squares
            r'\d+³',  # Cubes
            r'sin|cos|tan|log|ln|exp',  # Functions
            r'\∫|\∑|\∏',  # Advanced operations
            r'[a-zA-Z]\s*=\s*[a-zA-Z0-9\+\-\*\/]+',  # Equations
        ]
    
    def is_math_expression(self, text: str) -> bool:
        """Determine if text contains mathematical expressions"""
        text_clean = re.sub(r'\s+', ' ', text.strip())
        
        # Check for math patterns
        for pattern in self.math_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Check for math symbols
        math_symbols = ['=', '+', '-', '*', '/', '^', '√', '²', '³', '∫', '∑', '∏']
        symbol_count = sum(1 for symbol in math_symbols if symbol in text_clean)
        
        # If more than 20% of non-space characters are math symbols
        non_space_chars = len(text_clean.replace(' ', ''))
        if non_space_chars > 0 and symbol_count / non_space_chars > 0.2:
            return True
        
        return False
    
    def extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text"""
        expressions = []
        
        # Split by lines and sentences
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if self.is_math_expression(line):
                expressions.append(line)
            else:
                # Try to find math expressions within the line
                sentences = re.split(r'[.!?]', line)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if self.is_math_expression(sentence):
                        expressions.append(sentence)
        
        return expressions
    
    def process_image_data(self, image_data: str) -> Dict:
        """Process base64 image data"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get image info
            info = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "has_transparency": image.mode in ('RGBA', 'LA'),
            }
            
            return {
                "success": True,
                "image": image,
                "info": info
            }
            
        except Exception as e:
            logger.error(f"Error processing image data: {e}")
            return {
                "success": False,
                "error": str(e)
            }

class TranslationService:
    """Translation service utilities"""
    
    def __init__(self):
        self.language_codes = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'chinese': 'zh',
            'japanese': 'ja', 'korean': 'ko', 'arabic': 'ar', 'hindi': 'hi',
            'dutch': 'nl', 'swedish': 'sv', 'norwegian': 'no', 'danish': 'da',
            'finnish': 'fi', 'polish': 'pl', 'turkish': 'tr', 'greek': 'el',
            'hebrew': 'he', 'thai': 'th', 'vietnamese': 'vi', 'indonesian': 'id',
            'malay': 'ms', 'filipino': 'tl', 'ukrainian': 'uk', 'czech': 'cs',
            'slovak': 'sk', 'hungarian': 'hu', 'romanian': 'ro', 'bulgarian': 'bg',
            'croatian': 'hr', 'serbian': 'sr', 'slovenian': 'sl', 'estonian': 'et',
            'latvian': 'lv', 'lithuanian': 'lt'
        }
    
    def get_language_code(self, language: str) -> str:
        """Get language code from language name"""
        language_lower = language.lower()
        return self.language_codes.get(language_lower, language)
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        # This is a very basic implementation
        # In production, use a proper language detection library
        
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'ru'  # Russian
        elif re.search(r'[α-ωΑ-Ω]', text):
            return 'el'  # Greek
        elif re.search(r'[א-ת]', text):
            return 'he'  # Hebrew
        elif re.search(r'[ا-ي]', text):
            return 'ar'  # Arabic
        elif re.search(r'[一-龯]', text):
            return 'zh'  # Chinese
        elif re.search(r'[ひらがなカタカナ]', text):
            return 'ja'  # Japanese
        elif re.search(r'[가-힣]', text):
            return 'ko'  # Korean
        elif re.search(r'[ก-๙]', text):
            return 'th'  # Thai
        elif re.search(r'[ñáéíóúü]', text, re.IGNORECASE):
            return 'es'  # Spanish
        elif re.search(r'[àâäéèêëíîïóôöúùûü]', text, re.IGNORECASE):
            return 'fr'  # French
        elif re.search(r'[äöüß]', text, re.IGNORECASE):
            return 'de'  # German
        else:
            return 'en'  # Default to English

class GestureRecognizer:
    """Hand gesture recognition utilities"""
    
    def __init__(self):
        self.gesture_patterns = {
            'thumbs_up': 'Thumbs up - positive gesture',
            'thumbs_down': 'Thumbs down - negative gesture',
            'peace': 'Peace sign - victory or peace gesture',
            'ok': 'OK sign - approval gesture',
            'pointing': 'Pointing - indicating direction',
            'open_hand': 'Open hand - stop or greeting gesture',
            'fist': 'Closed fist - strength or agreement',
            'wave': 'Waving - greeting or goodbye'
        }
    
    def interpret_landmarks(self, landmarks) -> Dict:
        """Interpret hand landmarks to recognize gestures"""
        # This is a simplified implementation
        # In a real application, you would analyze the 21 hand landmarks
        # to determine finger positions and hand configuration
        
        # For demonstration, return a random gesture
        import random
        gesture = random.choice(list(self.gesture_patterns.keys()))
        
        return {
            "gesture": gesture,
            "description": self.gesture_patterns[gesture],
            "confidence": 0.7 + random.random() * 0.3
        }

class EmotionAnalyzer:
    """Emotion analysis from facial landmarks"""
    
    def __init__(self):
        self.emotions = {
            'happy': 'Happy - showing joy or contentment',
            'sad': 'Sad - showing sorrow or melancholy',
            'angry': 'Angry - showing irritation or frustration',
            'surprised': 'Surprised - showing shock or amazement',
            'neutral': 'Neutral - calm or expressionless',
            'confused': 'Confused - showing uncertainty',
            'focused': 'Focused - showing concentration'
        }
    
    def analyze_expression(self, blendshapes) -> Dict:
        """Analyze facial expression from MediaPipe blendshapes"""
        # This would normally analyze the blendshape coefficients
        # to determine the dominant emotion
        
        # For demonstration, return a weighted emotion
        import random
        emotion_weights = {
            'happy': 0.3,
            'neutral': 0.25,
            'focused': 0.2,
            'surprised': 0.1,
            'confused': 0.1,
            'sad': 0.03,
            'angry': 0.02
        }
        
        emotion = random.choices(
            list(emotion_weights.keys()),
            weights=list(emotion_weights.values())
        )[0]
        
        return {
            "emotion": emotion,
            "description": self.emotions[emotion],
            "confidence": 0.6 + random.random() * 0.4
        }