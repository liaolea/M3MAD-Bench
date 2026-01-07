from .cot import CoT
from .llm_debate import LLM_Debate_Main
from .dmad import DMAD_Main
from .io import IO_Main
from .self_consistency import SelfConsistency
from .div_mad import DivMADMain

method2class = {
    "cot": CoT,
    "llm_debate": LLM_Debate_Main,
    "div_mad": DivMADMain,
    "dmad": DMAD_Main,
    "io": IO_Main,
    "self_consistency": SelfConsistency,
}

def get_method_class(method_name, dataset_name=None):
    
    # lowercase the method name
    method_name = method_name.lower()
    
    all_method_names = method2class.keys()
    matched_method_names = [sample_method_name for sample_method_name in all_method_names if method_name in sample_method_name]
    
    if len(matched_method_names) > 0:
        if dataset_name is not None:
            # lowercase the dataset name
            dataset_name = dataset_name.lower()
            # check if there are method names that contain the dataset name
            matched_method_data_names = [sample_method_name for sample_method_name in matched_method_names if sample_method_name.split('_')[-1] in dataset_name]
            if len(matched_method_data_names) > 0:
                method_name = matched_method_data_names[0]
                if len(matched_method_data_names) > 1:
                    print(f"[WARNING] Found multiple methods matching {dataset_name}: {matched_method_data_names}. Using {method_name} instead.")
    else:
        raise ValueError(f"[ERROR] No method found matching {method_name}. Please check the method name.")
    
    return method2class[method_name]
