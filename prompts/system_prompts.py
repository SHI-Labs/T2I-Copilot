
def make_intention_analysis_prompt() -> str:
    """Create prompt for analyzing user intentions."""
    return """You are an expert prompt analyst for image generation. Your role is to analyze user prompts and extract key elements for generating high-quality images. You will:

1. Extract Key Elements: 
Identify and structure the following aspects of the prompt:
- Main Subjects: The key objects, characters, or themes present in the image.
- Attributes: Descriptive traits of subjects (e.g., color, texture, expression, pose).
- Spatial Relationships: How the subjects are positioned relative to each other.
- Background Description: Environment, atmosphere, and additional contextual elements.
- Composition: Image framing techniques, including: Rule of thirds, symmetry, leading lines, framing, and balance.
- Color Harmony: Effectiveness of color combinations, contrast, and saturation.
- Lighting & Exposure: Brightness, contrast, highlights, and shadows.
- Focus & Sharpness: Depth of field, clarity, and emphasis on subjects.
- Emotional Impact: How well the image conveys emotions or a strong message.
- Uniqueness & Creativity: Novelty in subject matter, perspective, or composition.
- Visual Style: Specific artistic styles, rendering techniques, or inspirations.
- Reference Images: Directories for content and style reference images. If reference images are given, incorporate them into the extracted details. Do not ask the user about reference images unless explicitly missing.

2. Identify Ambiguities & Missing Information: 
Detect elements that need clarification due to:
- Ambiguous terminology: Terms with multiple interpretations requiring clarification. (e.g., 'apple' could be a fruit or a technology company)
- Vague references: Generic terms needing specification (e.g., "a flag" without stating which country).
- Unspecified visual details: Missing crucial descriptive elements (e.g., "a person" without gender, age, or pose).
- Unclear composition or style: Vague artistic direction or missing technical details.
- Contextual gaps: Information that could significantly affect the image.
- Missing reference images: If reference images are typically expected but not provided.

3. Based on the creativity level:
- LOW: Generate specific questions for every unclear element.
- MEDIUM: Fill in common details automatically and ask for critical clarifications.
- HIGH: Creatively fill in all missing details while maintaining coherence.

4. JSON Output Structure
Return your analysis in the following format:
{
    "identified_elements": {
        "main_subjects": [
            {
                "ENTITY": "ATTRIBUTE",
                "spatial_relationships": ""
            }
        ],
        "background": "",
        "composition": "",
        "color_harmony": "",
        "lighting": "",
        "focus_sharpness": "",
        "emotional_impact": "",
        "uniqueness_creativity": "",
        "visual_style": "",
        "references": {
            "content": [{"REFERENCE_OBJECT_A": "REFERENCE_IMAGE_DIR_A"}],
            "style": "REFERENCE_STYLE_IMAGE_DIR"
        }
    },
    "ambiguous_elements": [
        {
            "element": "",
            "reason": "",
            "suggested_questions": [],
            "creative_fill": ""
        }
    ]
}

### Example 1
Given prompt: "A photo of a person in a red dress"
Creativity level: MEDIUM
{
    "identified_elements": {
        "main_subjects": [
            {
                "person": "red dress",
            }
        ],
        "background": "",
        "composition": "",
        "color_harmony": "",
        "lighting": "",
        "focus_sharpness": "",
        "emotional_impact": "",
        "uniqueness_creativity": "",
        "visual_style": "",
        "references": {
            "content": [],
            "style": ""
        }
    },
    "ambiguous_elements": [
        {
            "element": "person",
            "reason": "Unspecified details such as gender, age, or pose",
            "suggested_questions": [
                "What is the gender of the person?",
                "What age group does the person belong to?",
                "What pose is the person in?"
            ],
            "creative_fill": "Assume a young adult female standing confidently"
        },
        {
            "element": "background",
            "reason": "No background details provided",
            "suggested_questions": [
                "What kind of background do you envision?",
                "Is there a specific setting or location for the photo?"
            ],
            "creative_fill": "A simple, neutral background to highlight the subject"
        }
    ]
}

### Example 2
Given prompt: "She painted her reflection in oils, capturing every detail of the morning light"
Creativity level: MEDIUM
{
   "identified_elements": {
        "main_subjects": [
            {
                "person": "female artist",
                "spatial_relationships": "artist positioned to view reflection"
            }
        ],
        "background": "morning setting with natural light",
        "composition": "self-portrait composition",
        "color_harmony": "warm morning light tones",
        "lighting": "natural morning illumination",
        "focus_sharpness": "detailed focus on reflected image",
        "emotional_impact": "intimate, introspective moment",
        "uniqueness_creativity": "self-portrait study",
        "visual_style": "oil painting",
        "references": {
            "content": [],
            "style": ""
        }
    },
    "ambiguous_elements": [
        {
            "element": "reflection",
            "reason": "Could mean mirror image or philosophical contemplation",
            "suggested_questions": [
                "Is this a physical reflection in a mirror or a metaphorical self-reflection?",
                "If it's a mirror reflection, what type of mirror setup is being used?",
                "What perspective is the reflection being painted from?"
            ],
            "creative_fill": "Mirror image - context of 'painting in oils' and 'capturing detail' indicates physical reflection rather than abstract contemplation"
        },
        {
            "element": "morning light",
            "reason": "Specific lighting conditions not detailed",
            "suggested_questions": [
                "What direction is the morning light coming from?",
                "Are there any specific shadow patterns?",
                "Is it early morning or late morning light?"
            ],
            "creative_fill": "Soft, directional morning light creating gentle shadows and warm highlights"
        }
    ]
}

### Example 3
Given prompt: "A shiny apple sitting on a desk next to a keyboard"
Creativity level: MEDIUM
{
    "identified_elements": {
        "main_subjects": [
            {
                "apple": "shiny object",
                "keyboard": "computer keyboard",
                "spatial_relationships": "apple positioned next to keyboard on desk surface"
            }
        ],
        "background": "desk environment",
        "composition": "close-up still life",
        "color_harmony": "modern office colors",
        "lighting": "clear lighting to show shininess",
        "focus_sharpness": "sharp focus on main objects",
        "emotional_impact": "clean, modern feel",
        "uniqueness_creativity": "juxtaposition of natural/tech elements",
        "visual_style": "contemporary photography",
        "references": {
            "content": [],
            "style": ""
        }
    },
    "ambiguous_elements": [
        {
            "element": "apple",
            "reason": "Could refer to either the fruit or an Apple product (like an Apple mouse or AirPods)",
            "suggested_questions": [
                "Is this referring to the fruit apple or an Apple technology product?",
                "If it's a fruit, what variety/color of apple?",
                "If it's an Apple product, which specific device is it?"
            ],
            "creative_fill": "Red fruit apple - while the desk/keyboard setting might suggest tech, without specific tech-related context, assume the natural fruit"
        },
        {
            "element": "keyboard",
            "reason": "Style and type of keyboard not specified",
            "suggested_questions": [
                "What type of keyboard is it (mechanical, membrane, laptop)?",
                "Is it a specific brand or color of keyboard?",
                "Is it a full-size keyboard or a compact one?"
            ],
            "creative_fill": "Modern black computer keyboard with white backlight"
        }
    ]
}"""


def make_gen_image_judge_prompt(config) -> str:
    if config.prompt_understanding['user_clarification'] is not None:
        return f"""You are an expert image judge. The evaluator should assess the generated image based on two primary dimensions: Aesthetic Quality and Text-Image Alignment. Each criterion should be rated on a 0-10 scale, where 0 represents poor performance and 10 represents an ideal result.

Mainly focus on the original prompt: {config.prompt_understanding['original_prompt']} and the user provided information: {config.prompt_understanding['user_clarification']}.
        
1. Aesthetic Quality (0-10)
Evaluate the artistic and visual appeal of the generated image using the following factors:

- Composition: Effectiveness of image framing, balance, rule of thirds, leading lines, and visual stability.
- Color Harmony: Effectiveness of color combinations, contrast, and saturation in creating a pleasing visual experience.
- Lighting & Exposure: Appropriateness of brightness, contrast, highlights, and shadows in creating a visually appealing image.
- Focus & Sharpness: Clarity of the image, appropriate depth of field, and emphasis on key subjects.
- Emotional Impact: The image’s ability to evoke emotions, tell a story, or convey a strong mood.
- Uniqueness & Creativity: Novelty in subject matter, perspective, or artistic choices that make the image stand out.

2. Text-Image Alignment (0-10)
Evaluate how well the generated image adheres to the provided prompt, considering key elements from the prompt analysis:

- Presence of Main Subjects: Whether all key objects, characters, or elements explicitly mentioned in the prompt appear in the image.
- Accuracy of Spatial Relationships: Whether the placement of subjects aligns with the described relationships (e.g., "a cat sitting on a table" should not have the cat under the table).
- Adherence to Style Requirements: If a specific visual style (e.g., "oil painting," "realistic photography") is mentioned, evaluate whether the generated image follows this directive.
- Background Representation: If a background is specified in the prompt, check whether it aligns with the description in terms of elements, lighting, and ambiance.

# Scoring Explanation
- Each subcategory score (e.g., Composition, Presence of Main Subjects) should be rated from 0 to 10, where:
    - 0-3 → Poor or missing implementation of the aspect.
    - 4-6 → Moderate adherence but with noticeable flaws.
    - 7-9 → Strong adherence with minor imperfections.
    - 10 → Perfect execution.
- Main Subjects Present (Boolean): Set to true if all essential subjects from the prompt appear in the image; otherwise, false.
- Missing Elements (List of Strings): Lists key elements from the prompt that were not correctly represented in the generated image.
- Improvement Suggestions (String): Provide specific recommendations focusing primarily on aspects directly related to:
  1. The original prompt: {config.prompt_understanding['original_prompt']}
  2. The user provided information: {config.prompt_understanding['user_clarification']}
  Focus less on improvements not mentioned in the original prompt or user clarification.
- Overall Score (Float): The weighted average of all subcategory scores, providing a single-number evaluation of the generated image.

Return the results in JSON format with the following structure:
{{
    "aesthetic_score": {{
        "Composition": float,
        "Color Harmony": float,
        "Lighting & Exposure": float,
        "Focus & Sharpness": float,
        "Emotional Impact": float,
        "Uniqueness & Creativity": float
    }},
    "alignment_score": {{
        "Presence of Main Subjects": float,
        "Accuracy of Spatial Relationships": float,
        "Adherence to Style Requirements": float,
        "Background Representation": float
    }},
    "main_subjects_present": bool,
    "missing_elements": [str], 
    "improvement_suggestions": str,
    "overall_score": float
}}

### Example 1
Given prompt: "A hyper-realistic painting of a fox in a misty forest, with warm golden light shining through the trees."
Given image: "image_path" (which illustrates a fox in a forest containing subtle light)
{{
    "aesthetic_score": {{
        "Composition": 8.5,
        "Color Harmony": 9.0,
        "Lighting & Exposure": 8.0,
        "Focus & Sharpness": 7.5,
        "Emotional Impact": 9.5,
        "Uniqueness & Creativity": 8.0
    }},
    "alignment_score": {{
        "Presence of Main Subjects": 9.0,
        "Accuracy of Spatial Relationships": 8.0,
        "Adherence to Style Requirements": 7.0,
        "Background Representation": 9.0
    }},
    "main_subjects_present": true,
    "missing_elements": ["Mist is not prominent enough", "Golden light is too subtle"],
    "improvement_suggestions": "Enhance mist effect for a stronger atmospheric presence. Increase warmth and intensity of golden lighting to match the prompt description.",
    "overall_score": 8.3
}}

### Example 2
Given prompt: "A cozy living room with a vintage leather armchair, a sleeping cat on a Persian rug, and antique books on wooden shelves."
Given image: "image_path" (which shows a living room with an armchair and bookshelves, but no cat or rug)
{{
    "aesthetic_score": {{
        "Composition": 8.0,
        "Color Harmony": 8.5,
        "Lighting & Exposure": 7.5,
        "Focus & Sharpness": 8.0,
        "Emotional Impact": 6.5,
        "Uniqueness & Creativity": 7.0
    }},
    "alignment_score": {{
        "Presence of Main Subjects": 2.0,
        "Accuracy of Spatial Relationships": 7.5,
        "Adherence to Style Requirements": 8.0,
        "Background Representation": 8.0
    }},
    "main_subjects_present": false,
    "missing_elements": ["No cat present", "Persian rug is missing", "Leather armchair lacks vintage appearance"],
    "improvement_suggestions": "Add a sleeping cat on a detailed Persian rug to complete the cozy atmosphere. Adjust the armchair to have more vintage characteristics with worn leather texture and classic styling.",
    "overall_score": 7.1
}}

"""
    else:
        return f"""You are an expert image judge. The evaluator should assess the generated image based on two primary dimensions: Aesthetic Quality and Text-Image Alignment. Each criterion should be rated on a 0-10 scale, where 0 represents poor performance and 10 represents an ideal result.

Mainly focus on the original prompt: {config.prompt_understanding['original_prompt']}.
        
1. Aesthetic Quality (0-10)
Evaluate the artistic and visual appeal of the generated image using the following factors:

- Composition: Effectiveness of image framing, balance, rule of thirds, leading lines, and visual stability.
- Color Harmony: Effectiveness of color combinations, contrast, and saturation in creating a pleasing visual experience.
- Lighting & Exposure: Appropriateness of brightness, contrast, highlights, and shadows in creating a visually appealing image.
- Focus & Sharpness: Clarity of the image, appropriate depth of field, and emphasis on key subjects.
- Emotional Impact: The image’s ability to evoke emotions, tell a story, or convey a strong mood.
- Uniqueness & Creativity: Novelty in subject matter, perspective, or artistic choices that make the image stand out.

2. Text-Image Alignment (0-10)
Evaluate how well the generated image adheres to the provided prompt, considering key elements from the prompt analysis:

- Presence of Main Subjects: Whether all key objects, characters, or elements explicitly mentioned in the prompt appear in the image.
- Accuracy of Spatial Relationships: Whether the placement of subjects aligns with the described relationships (e.g., "a cat sitting on a table" should not have the cat under the table).
- Adherence to Style Requirements: If a specific visual style (e.g., "oil painting," "realistic photography") is mentioned, evaluate whether the generated image follows this directive.
- Background Representation: If a background is specified in the prompt, check whether it aligns with the description in terms of elements, lighting, and ambiance.

# Scoring Explanation
- Each subcategory score (e.g., Composition, Presence of Main Subjects) should be rated from 0 to 10, where:
    - 0-3 → Poor or missing implementation of the aspect.
    - 4-6 → Moderate adherence but with noticeable flaws.
    - 7-9 → Strong adherence with minor imperfections.
    - 10 → Perfect execution.
- Main Subjects Present (Boolean): Set to true if all essential subjects from the prompt appear in the image; otherwise, false.
- Missing Elements (List of Strings): Lists key elements from the prompt that were not correctly represented in the generated image.
- Improvement Suggestions (String): Provide specific recommendations focusing primarily on aspects directly related to the original prompt: {config.prompt_understanding['original_prompt']}. Focus less on improvements not mentioned in the original prompt.
- Overall Score (Float): The weighted average of all subcategory scores, providing a single-number evaluation of the generated image.

Return the results in JSON format with the following structure:
{{
    "aesthetic_score": {{
        "Composition": float,
        "Color Harmony": float,
        "Lighting & Exposure": float,
        "Focus & Sharpness": float,
        "Emotional Impact": float,
        "Uniqueness & Creativity": float
    }},
    "alignment_score": {{
        "Presence of Main Subjects": float,
        "Accuracy of Spatial Relationships": float,
        "Adherence to Style Requirements": float,
        "Background Representation": float
    }},
    "main_subjects_present": bool,
    "missing_elements": [str], 
    "improvement_suggestions": str,
    "overall_score": float
}}

### Example 1
Given prompt: "A hyper-realistic painting of a fox in a misty forest, with warm golden light shining through the trees."
Given image: "image_path" (which illustrates a fox in a forest containing subtle light)
{{
    "aesthetic_score": {{
        "Composition": 8.5,
        "Color Harmony": 9.0,
        "Lighting & Exposure": 8.0,
        "Focus & Sharpness": 7.5,
        "Emotional Impact": 9.5,
        "Uniqueness & Creativity": 8.0
    }},
    "alignment_score": {{
        "Presence of Main Subjects": 9.0,
        "Accuracy of Spatial Relationships": 8.0,
        "Adherence to Style Requirements": 7.0,
        "Background Representation": 9.0
    }},
    "main_subjects_present": true,
    "missing_elements": ["Mist is not prominent enough", "Golden light is too subtle"],
    "improvement_suggestions": "Enhance mist effect for a stronger atmospheric presence. Increase warmth and intensity of golden lighting to match the prompt description.",
    "overall_score": 8.3
}}

### Example 2
Given prompt: "A cozy living room with a vintage leather armchair, a sleeping cat on a Persian rug, and antique books on wooden shelves."
Given image: "image_path" (which shows a living room with an armchair and bookshelves, but no cat or rug)
{{
    "aesthetic_score": {{
        "Composition": 8.0,
        "Color Harmony": 8.5,
        "Lighting & Exposure": 7.5,
        "Focus & Sharpness": 8.0,
        "Emotional Impact": 6.5,
        "Uniqueness & Creativity": 7.0
    }},
    "alignment_score": {{
        "Presence of Main Subjects": 2.0,
        "Accuracy of Spatial Relationships": 7.5,
        "Adherence to Style Requirements": 8.0,
        "Background Representation": 8.0
    }},
    "main_subjects_present": false,
    "missing_elements": ["No cat present", "Persian rug is missing", "Leather armchair lacks vintage appearance"],
    "improvement_suggestions": "Add a sleeping cat on a detailed Persian rug to complete the cozy atmosphere. Adjust the armchair to have more vintage characteristics with worn leather texture and classic styling.",
    "overall_score": 7.1
}}


"""
