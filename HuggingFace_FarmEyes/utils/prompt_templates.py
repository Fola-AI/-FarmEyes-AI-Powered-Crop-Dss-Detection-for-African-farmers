"""
FarmEyes Prompt Templates
=========================
Contains all prompt templates for N-ATLaS language model interactions.
Includes templates for translation, disease diagnosis reports, and
multilingual communication in Hausa, Yoruba, Igbo, and English.

N-ATLaS Model: tosinamuda/N-ATLaS-GGUF (8B parameters)
Supported Languages: English (en), Hausa (ha), Yoruba (yo), Igbo (ig)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# LANGUAGE MAPPINGS
# =============================================================================

# Language code to full name mapping
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo"
}

# Language code to native name mapping (for display)
LANGUAGE_NATIVE_NAMES: Dict[str, str] = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yorùbá",
    "ig": "Asụsụ Igbo"
}


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

@dataclass
class SystemPrompts:
    """System prompts for different N-ATLaS tasks"""
    
    # General agricultural assistant system prompt
    AGRICULTURAL_ASSISTANT: str = """You are FarmEyes, an AI agricultural assistant designed to help Nigerian farmers.
Your role is to provide clear, practical advice about crop diseases and treatments.
Always communicate in a respectful, helpful manner using simple language that farmers can understand.
When discussing costs, always use Nigerian Naira (₦).
Focus on actionable advice that farmers can implement with locally available resources."""

    # Translation system prompt
    TRANSLATOR: str = """You are a professional translator specializing in Nigerian languages.
Your task is to translate agricultural content accurately while maintaining clarity and cultural appropriateness.
Translate naturally - do not translate word-by-word. Ensure the meaning is preserved and easily understood by farmers.
Keep technical terms simple and use local equivalents where possible."""

    # Disease diagnosis system prompt
    DISEASE_DIAGNOSIS: str = """You are FarmEyes, an expert agricultural AI assistant helping Nigerian farmers identify and treat crop diseases.
Provide comprehensive but easy-to-understand diagnosis reports.
Include practical treatment options with local costs in Nigerian Naira (₦).
Mention both modern treatments and traditional methods where applicable.
Always prioritize the farmer's safety when recommending chemical treatments."""

    # Conversational assistant system prompt
    CONVERSATIONAL: str = """You are FarmEyes, a friendly AI farming assistant.
Help Nigerian farmers with their questions about crop diseases, treatments, and farming practices.
Respond in the same language the farmer uses.
Be patient, respectful, and provide practical advice.
If you don't know something, say so honestly and suggest consulting a local agricultural extension officer."""


# =============================================================================
# TRANSLATION PROMPTS
# =============================================================================

class TranslationPrompts:
    """Prompts for translating content between languages"""
    
    @staticmethod
    def translate_text(text: str, target_language: str) -> str:
        """
        Generate a prompt to translate text to target language.
        
        Args:
            text: The text to translate
            target_language: Target language code (ha, yo, ig, en)
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Translate the following text to {lang_name}. 
Provide only the translation, no explanations or additional text.
Maintain the same tone and meaning. Use simple, clear language that farmers can understand.

Text to translate:
{text}

{lang_name} translation:"""

    @staticmethod
    def translate_disease_name(disease_name: str, target_language: str) -> str:
        """
        Generate a prompt to translate a disease name.
        
        Args:
            disease_name: Name of the disease in English
            target_language: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Translate this crop disease name to {lang_name}.
If there is a commonly used local name for this disease, use that.
If not, translate it descriptively so farmers understand what it is.
Provide only the translation.

Disease name: {disease_name}

{lang_name} name:"""

    @staticmethod
    def translate_symptoms(symptoms: List[str], target_language: str) -> str:
        """
        Generate a prompt to translate a list of symptoms.
        
        Args:
            symptoms: List of symptom descriptions
            target_language: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        symptoms_text = "\n".join([f"- {s}" for s in symptoms])
        
        return f"""Translate these crop disease symptoms to {lang_name}.
Keep each symptom as a separate line starting with a dash (-).
Use simple language that farmers can easily understand.

Symptoms:
{symptoms_text}

{lang_name} translation:"""

    @staticmethod
    def translate_treatment(treatment: str, target_language: str) -> str:
        """
        Generate a prompt to translate treatment instructions.
        
        Args:
            treatment: Treatment instructions in English
            target_language: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Translate these treatment instructions to {lang_name}.
Keep the instructions clear and easy to follow.
Maintain any measurements, dosages, and costs exactly as given.
Use simple language that farmers can understand.

Treatment instructions:
{treatment}

{lang_name} translation:"""

    @staticmethod
    def translate_ui_text(text: str, target_language: str) -> str:
        """
        Generate a prompt to translate UI text (buttons, labels, etc.).
        
        Args:
            text: UI text to translate
            target_language: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Translate this app interface text to {lang_name}.
Keep it concise and natural. This is for a button/label in a mobile app.
Provide only the translation.

Text: {text}

{lang_name}:"""

    @staticmethod
    def batch_translate(texts: List[str], target_language: str) -> str:
        """
        Generate a prompt to translate multiple texts at once.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        numbered_texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
        
        return f"""Translate each of the following texts to {lang_name}.
Keep the same numbering format. Provide only the translations.

Texts to translate:
{numbered_texts}

{lang_name} translations:"""


# =============================================================================
# DISEASE DIAGNOSIS PROMPTS
# =============================================================================

class DiagnosisPrompts:
    """Prompts for generating disease diagnosis reports"""
    
    @staticmethod
    def generate_diagnosis_summary(
        disease_name: str,
        crop: str,
        confidence: float,
        severity: str,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for creating a diagnosis summary.
        
        Args:
            disease_name: Name of the detected disease
            crop: Type of crop (cassava, cocoa, tomato)
            confidence: Detection confidence (0.0 - 1.0)
            severity: Severity level (low, medium, high, very_high)
            target_language: Language for the summary
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        confidence_percent = int(confidence * 100)
        
        return f"""Generate a brief diagnosis summary in {lang_name} for a Nigerian farmer.

Disease detected: {disease_name}
Crop: {crop}
Detection confidence: {confidence_percent}%
Severity: {severity}

Write a 2-3 sentence summary that:
1. Tells the farmer what disease was found
2. Indicates how serious it is
3. Reassures them that treatment options are available

Use simple, clear language. Be direct but compassionate.

{lang_name} summary:"""

    @staticmethod
    def generate_treatment_recommendation(
        disease_name: str,
        treatments: Dict,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for treatment recommendations.
        
        Args:
            disease_name: Name of the disease
            treatments: Dictionary containing treatment information
            target_language: Language for recommendations
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        # Format treatment info
        treatment_text = f"Disease: {disease_name}\n\n"
        
        if "cultural" in treatments:
            treatment_text += "Cultural practices:\n"
            for t in treatments["cultural"][:3]:  # Limit to top 3
                treatment_text += f"- {t.get('method', '')}: {t.get('description', '')}\n"
        
        if "chemical" in treatments:
            treatment_text += "\nChemical treatments:\n"
            for t in treatments["chemical"][:2]:  # Limit to top 2
                treatment_text += f"- {t.get('product_name', '')}: {t.get('dosage', '')}\n"
                if "cost_ngn_min" in t:
                    treatment_text += f"  Cost: ₦{t['cost_ngn_min']:,} - ₦{t.get('cost_ngn_max', t['cost_ngn_min']):,}\n"
        
        if "traditional" in treatments:
            treatment_text += "\nTraditional methods:\n"
            for t in treatments["traditional"][:2]:
                treatment_text += f"- {t.get('method', '')}: {t.get('description', '')}\n"
        
        return f"""Based on this treatment information, provide practical recommendations in {lang_name}.

{treatment_text}

Write treatment advice that:
1. Starts with the most important immediate action
2. Lists 3-4 specific treatment options
3. Includes estimated costs in Nigerian Naira (₦)
4. Mentions safety precautions for chemical treatments
5. Uses simple language a farmer can understand

{lang_name} treatment recommendations:"""

    @staticmethod
    def generate_prevention_advice(
        disease_name: str,
        prevention_tips: List[str],
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for prevention advice.
        
        Args:
            disease_name: Name of the disease
            prevention_tips: List of prevention methods
            target_language: Language for advice
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        tips_text = "\n".join([f"- {tip}" for tip in prevention_tips[:6]])
        
        return f"""Provide prevention advice in {lang_name} for this crop disease.

Disease: {disease_name}

Prevention methods:
{tips_text}

Write practical prevention advice that:
1. Explains the most important prevention steps
2. Uses simple language farmers can understand
3. Focuses on actions they can take immediately
4. Mentions low-cost options first

{lang_name} prevention advice:"""

    @staticmethod
    def generate_health_projection(
        disease_name: str,
        infection_stage: str,
        recovery_chance: int,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for health projection message.
        
        Args:
            disease_name: Name of the disease
            infection_stage: Stage of infection (early, moderate, severe)
            recovery_chance: Percentage chance of recovery
            target_language: Language for the message
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Generate a health projection message in {lang_name} for a farmer.

Disease: {disease_name}
Infection stage: {infection_stage}
Recovery chance: {recovery_chance}%

Write a message that:
1. Is honest but hopeful (if there's a reasonable chance)
2. Tells the farmer what to expect
3. Encourages immediate action
4. Uses simple, compassionate language

{lang_name} message:"""

    @staticmethod
    def generate_healthy_plant_message(
        crop: str,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for healthy plant detection message.
        
        Args:
            crop: Type of crop
            target_language: Language for message
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Generate a positive message in {lang_name} for a farmer whose {crop} plant is healthy.

Write a message that:
1. Congratulates them on having a healthy plant
2. Gives 2-3 tips to maintain plant health
3. Encourages regular monitoring
4. Is warm and encouraging

{lang_name} message:"""


# =============================================================================
# FULL DIAGNOSIS REPORT PROMPT
# =============================================================================

class ReportPrompts:
    """Prompts for generating complete diagnosis reports"""
    
    @staticmethod
    def generate_full_report(
        disease_data: Dict,
        confidence: float,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt for a complete diagnosis report.
        
        Args:
            disease_data: Full disease information from knowledge base
            confidence: Detection confidence (0.0 - 1.0)
            target_language: Language for the report
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        confidence_percent = int(confidence * 100)
        
        # Extract key information
        disease_name = disease_data.get("display_name", "Unknown Disease")
        crop = disease_data.get("crop", "crop")
        severity = disease_data.get("severity", {}).get("level", "unknown")
        symptoms = disease_data.get("symptoms", [])[:5]
        yield_loss = disease_data.get("yield_loss", {})
        
        symptoms_text = "\n".join([f"- {s}" for s in symptoms])
        yield_text = f"{yield_loss.get('min_percent', 0)}% - {yield_loss.get('max_percent', 100)}%"
        
        return f"""Generate a complete diagnosis report in {lang_name} for a Nigerian farmer.

DIAGNOSIS INFORMATION:
- Disease: {disease_name}
- Crop: {crop}
- Confidence: {confidence_percent}%
- Severity: {severity}
- Potential yield loss: {yield_text}

Key symptoms:
{symptoms_text}

Create a report with these sections:
1. DIAGNOSIS SUMMARY (2-3 sentences explaining what was found)
2. WHAT TO DO NOW (immediate actions, numbered list)
3. TREATMENT OPTIONS (brief mention of available treatments)
4. EXPECTED OUTCOME (what to expect with treatment)

Guidelines:
- Use simple, clear {lang_name}
- Be direct but compassionate
- Focus on practical actions
- Keep it concise (under 300 words)

{lang_name} Report:"""


# =============================================================================
# CONVERSATIONAL PROMPTS
# =============================================================================

class ConversationalPrompts:
    """Prompts for conversational interactions with farmers"""
    
    @staticmethod
    def answer_farmer_question(
        question: str,
        context: Optional[str] = None,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt to answer a farmer's question.
        
        Args:
            question: The farmer's question
            context: Optional context (e.g., current diagnosis)
            target_language: Language for response
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        context_section = ""
        if context:
            context_section = f"\nContext: {context}\n"
        
        return f"""You are FarmEyes, an AI farming assistant. Answer this farmer's question in {lang_name}.
{context_section}
Farmer's question: {question}

Provide a helpful, practical answer that:
1. Directly addresses their question
2. Uses simple language
3. Gives actionable advice where possible
4. Is respectful and encouraging

{lang_name} response:"""

    @staticmethod
    def clarify_treatment(
        treatment_name: str,
        treatment_info: str,
        target_language: str = "en"
    ) -> str:
        """
        Generate a prompt to explain a treatment in more detail.
        
        Args:
            treatment_name: Name of the treatment
            treatment_info: Treatment information
            target_language: Language for explanation
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Explain this treatment in simple {lang_name} that a farmer can easily understand.

Treatment: {treatment_name}
Information: {treatment_info}

Provide a clear explanation that:
1. Explains what this treatment does
2. How to apply it step by step
3. When to apply it
4. Any safety precautions
5. Where to buy it (agro-dealers)

{lang_name} explanation:"""

    @staticmethod
    def provide_encouragement(
        situation: str,
        target_language: str = "en"
    ) -> str:
        """
        Generate encouraging message for a farmer facing challenges.
        
        Args:
            situation: Description of farmer's situation
            target_language: Language for message
            
        Returns:
            Formatted prompt string
        """
        lang_name = LANGUAGE_NAMES.get(target_language, "English")
        
        return f"""Write an encouraging message in {lang_name} for a farmer in this situation:

{situation}

The message should:
1. Acknowledge their challenge
2. Offer hope and practical next steps
3. Be warm and supportive
4. Remind them that many farmers have overcome similar challenges

{lang_name} message:"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_system_prompt(task: str = "general") -> str:
    """
    Get the appropriate system prompt for a task.
    
    Args:
        task: Type of task (general, translation, diagnosis, conversation)
        
    Returns:
        System prompt string
    """
    prompts = SystemPrompts()
    
    task_mapping = {
        "general": prompts.AGRICULTURAL_ASSISTANT,
        "translation": prompts.TRANSLATOR,
        "diagnosis": prompts.DISEASE_DIAGNOSIS,
        "conversation": prompts.CONVERSATIONAL
    }
    
    return task_mapping.get(task, prompts.AGRICULTURAL_ASSISTANT)


def format_prompt_for_natlas(
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Format a prompt for N-ATLaS model using the expected chat format.
    
    Args:
        system_prompt: The system instruction
        user_prompt: The user's message/request
        
    Returns:
        Formatted prompt string for N-ATLaS
    """
    # N-ATLaS uses a chat format similar to Llama models
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def get_language_name(code: str, native: bool = False) -> str:
    """
    Get language name from code.
    
    Args:
        code: Language code (en, ha, yo, ig)
        native: If True, return native name
        
    Returns:
        Language name string
    """
    if native:
        return LANGUAGE_NATIVE_NAMES.get(code, "English")
    return LANGUAGE_NAMES.get(code, "English")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Generate a translation prompt
    print("=" * 60)
    print("Example Translation Prompt (English to Hausa)")
    print("=" * 60)
    
    text = "Your cassava plant has bacterial blight. Remove infected plants immediately."
    prompt = TranslationPrompts.translate_text(text, "ha")
    print(prompt)
    
    print("\n" + "=" * 60)
    print("Example Diagnosis Summary Prompt (Yoruba)")
    print("=" * 60)
    
    prompt = DiagnosisPrompts.generate_diagnosis_summary(
        disease_name="Cassava Mosaic Disease",
        crop="cassava",
        confidence=0.92,
        severity="high",
        target_language="yo"
    )
    print(prompt)
    
    print("\n" + "=" * 60)
    print("Example Formatted N-ATLaS Prompt")
    print("=" * 60)
    
    system = get_system_prompt("translation")
    user = TranslationPrompts.translate_text("Good morning, farmer!", "ig")
    formatted = format_prompt_for_natlas(system, user)
    print(formatted)
