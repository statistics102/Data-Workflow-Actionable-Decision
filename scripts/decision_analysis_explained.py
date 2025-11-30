# decision_analysis_explained.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

class DecisionAnalysis:
    def __init__(self):
        self.df = None
        self.tree_model = None
        self.feature_names = None
        self.cutoff_ranges = {}
        
    def load_data(self):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv('data/cleaned_analysis_data.csv')
            print("✅ Loaded cleaned data for decision analysis")
        except:
            print("🔄 Generating sample data...")
            from generate_sample_data import create_sample_data
            from complete_analysis_pipeline import DataAnalysisPipeline
            create_sample_data()
            pipeline = DataAnalysisPipeline()
            pipeline.load_data().data_cleaning()
            self.df = pipeline.cleaned_df
        
        self.feature_names = ['age', 'income', 'spending_score', 'credit_score', 'satisfaction']
        
        # Detailed target analysis
        self.analyze_target_classes()
        
        return self
    
    def analyze_target_classes(self):
        """Detailed analysis of target classes 1 and 0"""
        print("\n🎯 DETAILED TARGET CLASS ANALYSIS")
        print("=" * 50)
        
        target_counts = self.df['target'].value_counts()
        print(f"📊 Class Distribution:")
        print(f"   Class 0: {target_counts[0]} customers ({target_counts[0]/len(self.df)*100:.1f}%)")
        print(f"   Class 1: {target_counts[1]} customers ({target_counts[1]/len(self.df)*100:.1f}%)")
        
        # Analyze characteristics of each class
        print(f"\n📈 CHARACTERISTICS BY TARGET CLASS:")
        print("-" * 40)
        
        for class_val in [0, 1]:
            class_data = self.df[self.df['target'] == class_val]
            class_name = "CLASS 0 (Standard Customers)" if class_val == 0 else "CLASS 1 (High-Value Targets)"
            
            print(f"\n{class_name}:")
            print(f"   👥 Count: {len(class_data)} customers")
            print(f"   📊 Profile Summary:")
            print(f"      • Average Age: {class_data['age'].mean():.1f} years")
            print(f"      • Average Income: ${class_data['income'].mean():,.0f}")
            print(f"      • Average Spending Score: {class_data['spending_score'].mean():.1f}")
            print(f"      • Average Credit Score: {class_data['credit_score'].mean():.1f}")
            print(f"      • Average Satisfaction: {class_data['satisfaction'].mean():.1f}/5")
            
            # Key differentiators
            if class_val == 1:
                print(f"   💡 KEY DIFFERENTIATORS (vs Class 0):")
                class0_data = self.df[self.df['target'] == 0]
                for feature in self.feature_names:
                    diff = class_data[feature].mean() - class0_data[feature].mean()
                    if abs(diff) > 0:
                        direction = "higher" if diff > 0 else "lower"
                        print(f"      • {feature}: {abs(diff):.1f} {direction}")
    
    def train_and_analyze_tree(self, max_depth=4):
        """Train decision tree and analyze cutoff values"""
        print("\n🔍 TRAINING DECISION TREE AND ANALYZING CUTOFF VALUES")
        print("=" * 60)
        
        # Prepare data
        X = self.df[self.feature_names]
        y = self.df['target']
        
        # Train decision tree
        self.tree_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=20,
            random_state=42
        )
        self.tree_model.fit(X, y)
        
        # Extract tree structure for analysis
        self.analyze_tree_structure()
        self.explain_business_decisions()
        self.create_cutoff_visualization()
        
        return self
    
    def analyze_tree_structure(self):
        """Analyze the tree structure and extract all cutoff values"""
        print("\n📊 DECISION TREE CUTOFF ANALYSIS")
        print("=" * 50)
        
        tree_struct = self.tree_model.tree_
        feature_names = self.feature_names
        
        # Get all unique cutoff values for each feature
        for feature_idx in range(len(feature_names)):
            feature_name = feature_names[feature_idx]
            # Get all thresholds used for this feature
            thresholds = tree_struct.threshold[tree_struct.feature == feature_idx]
            unique_thresholds = np.unique(thresholds[thresholds != -2])  # -2 indicates leaf nodes
            
            if len(unique_thresholds) > 0:
                self.cutoff_ranges[feature_name] = {
                    'cutoffs': unique_thresholds,
                    'min_value': self.df[feature_name].min(),
                    'max_value': self.df[feature_name].max(),
                    'mean_value': self.df[feature_name].mean()
                }
                
                print(f"\n🎯 {feature_name.upper()}:")
                print(f"   Data Range: {self.cutoff_ranges[feature_name]['min_value']:.1f} to {self.cutoff_ranges[feature_name]['max_value']:.1f}")
                print(f"   Mean Value: {self.cutoff_ranges[feature_name]['mean_value']:.1f}")
                print(f"   Decision Cutoffs: {[f'{x:.1f}' for x in unique_thresholds]}")
                
                # Interpret what these cutoffs mean
                self.interpret_cutoffs(feature_name, unique_thresholds)
            else:
                print(f"\n📊 {feature_name.upper()}:")
                print(f"   No specific cutoffs found in this tree depth")
                print(f"   Data Range: {self.df[feature_name].min():.1f} to {self.df[feature_name].max():.1f}")
                print(f"   Mean Value: {self.df[feature_name].mean():.1f}")
        
        return self.cutoff_ranges
    
    def interpret_cutoffs(self, feature_name, cutoffs):
        """Interpret what the cutoff values mean in business terms"""
        interpretations = {
            'age': {
                'low': "Young customers (18-35)",
                'medium': "Middle-aged customers (36-50)", 
                'high': "Senior customers (51+)",
                'unit': 'years'
            },
            'income': {
                'low': "Low-income segment (<$45K)",
                'medium': "Middle-income segment ($45K-$65K)",
                'high': "High-income segment (>$65K)", 
                'unit': 'USD'
            },
            'spending_score': {
                'low': "Conservative spenders (<45)",
                'medium': "Moderate spenders (45-55)",
                'high': "High spenders (>55)",
                'unit': 'points'
            },
            'credit_score': {
                'low': "Poor credit risk (<620)",
                'medium': "Average credit (620-720)", 
                'high': "Excellent credit (>720)",
                'unit': 'points'
            },
            'satisfaction': {
                'low': "Dissatisfied customers (<3.0)",
                'medium': "Neutral satisfaction (3.0-4.0)",
                'high': "Highly satisfied (>4.0)",
                'unit': 'rating (1-5 scale)'
            }
        }
        
        if feature_name in interpretations:
            feat_info = interpretations[feature_name]
            print(f"   📝 Business Interpretation:")
            
            if len(cutoffs) == 1:
                cutoff = cutoffs[0]
                print(f"      • Below {cutoff:.1f} {feat_info['unit']}: {feat_info['low']}")
                print(f"      • Above {cutoff:.1f} {feat_info['unit']}: {feat_info['high']}")
            elif len(cutoffs) >= 2:
                print(f"      • Below {cutoffs[0]:.1f} {feat_info['unit']}: {feat_info['low']}")
                print(f"      • {cutoffs[0]:.1f}-{cutoffs[1]:.1f} {feat_info['unit']}: {feat_info['medium']}")
                print(f"      • Above {cutoffs[1]:.1f} {feat_info['unit']}: {feat_info['high']}")
    
    def explain_business_decisions(self):
        """Explain how decisions are made based on cutoff values"""
        print("\n🎯 HOW DECISIONS ARE MADE")
        print("=" * 50)
        
        # Extract tree rules in readable format
        tree_rules = export_text(self.tree_model, 
                               feature_names=self.feature_names,
                               decimals=1,
                               show_weights=True)
        
        print("📋 DECISION RULES EXTRACTED FROM TREE:")
        print("-" * 40)
        
        rules = tree_rules.split('\n')
        rule_number = 1
        current_conditions = []
        
        for i, rule in enumerate(rules):
            if '─' in rule and 'class' not in rule:
                # This is a condition line
                condition = rule.replace('─', '').replace('|', '').strip()
                if condition and 'class' not in condition:
                    current_conditions.append(condition)
            
            if 'class' in rule and 'samples' in rule:
                # This is a decision rule
                if current_conditions:  # Only show if we have conditions
                    print(f"\n📊 RULE {rule_number}:")
                    
                    # Show the key conditions that lead to this decision
                    for condition in current_conditions[-3:]:  # Show last 3 conditions
                        if condition:
                            print(f"   ✅ Condition: {condition}")
                    
                    # Extract class and samples
                    class_part = rule.split('class:')[-1].split('samples')[0].strip()
                    samples_part = rule.split('samples:')[-1].split('value')[0].strip()
                    value_part = rule.split('value:')[-1].strip()
                    
                    print(f"   🎯 Decision: Class {class_part}")
                    print(f"   📊 Confidence: Based on {samples_part} samples")
                    print(f"   📈 Distribution: {value_part}")
                    
                    # Business interpretation
                    self.interpret_decision_rule(current_conditions, int(class_part))
                    rule_number += 1
                
                current_conditions = []  # Reset for next rule
        
        print(f"\n🔍 Total Decision Paths Analyzed: {rule_number-1}")
        return self
    
    def interpret_decision_rule(self, conditions, predicted_class):
        """Provide business interpretation for a decision rule"""
        print(f"   💡 Business Insight:", end=" ")
        
        # Extract key conditions
        age_info = None
        income_info = None
        spending_info = None
        credit_info = None
        satisfaction_info = None
        
        for condition in conditions:
            if 'age' in condition:
                age_info = condition
            elif 'income' in condition:
                income_info = condition
            elif 'spending_score' in condition:
                spending_info = condition
            elif 'credit_score' in condition:
                credit_info = condition
            elif 'satisfaction' in condition:
                satisfaction_info = condition
        
        insights = []
        
        if age_info:
            if '<=' in age_info:
                age_val = float(age_info.split('<=')[-1])
                if age_val < 40:
                    insights.append("younger customers")
                else:
                    insights.append("middle-aged customers")
            else:
                insights.append("older customers")
        
        if income_info:
            if '<=' in income_info:
                income_val = float(income_info.split('<=')[-1])
                if income_val < 45000:
                    insights.append("lower income")
                else:
                    insights.append("moderate income")
            else:
                insights.append("higher income")
        
        if spending_info:
            if '<=' in spending_info:
                insights.append("conservative spending")
            else:
                insights.append("active spending")
        
        if credit_info:
            if '<=' in credit_info:
                insights.append("average credit")
            else:
                insights.append("good credit")
        
        if satisfaction_info:
            if '<=' in satisfaction_info:
                insights.append("average satisfaction")
            else:
                insights.append("high satisfaction")
        
        if predicted_class == 1:
            target_desc = "HIGH-VALUE TARGET"
            action = "Focus marketing efforts, offer premium services"
        else:
            target_desc = "STANDARD CUSTOMER" 
            action = "Maintain standard service, focus on retention"
        
        if insights:
            print(f"This {target_desc} typically represents {', '.join(insights)}.")
        else:
            print(f"This represents a {target_desc} segment.")
        
        print(f"   🚀 Recommended Action: {action}")
    
    def create_cutoff_visualization(self):
        """Create visualization showing cutoff values and their impact"""
        print("\n📈 CREATING CUTOFF VALUE VISUALIZATION")
        
        # Create subplots for each feature that has cutoffs
        features_with_cutoffs = [f for f in self.feature_names if f in self.cutoff_ranges]
        
        if not features_with_cutoffs:
            print("❌ No cutoff values found to visualize")
            return self
        
        n_features = len(features_with_cutoffs)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for i, feature in enumerate(features_with_cutoffs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot distribution by target class
            self.df[self.df['target'] == 0][feature].hist(
                bins=20, alpha=0.7, color='red', label='Class 0 (Standard)', ax=ax
            )
            self.df[self.df['target'] == 1][feature].hist(
                bins=20, alpha=0.7, color='blue', label='Class 1 (High-Value)', ax=ax
            )
            
            # Add cutoff lines
            for cutoff in self.cutoff_ranges[feature]['cutoffs']:
                ax.axvline(cutoff, color='green', linestyle='--', linewidth=2, 
                          label=f'Decision Cutoff: {cutoff:.1f}')
            
            ax.set_title(f'{feature.upper()} Distribution & Cutoffs', fontweight='bold', fontsize=12)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('DECISION CUTOFF ANALYSIS: How Variables Split Customer Segments', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('decision_cutoff_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Cutoff analysis visualization saved as 'decision_cutoff_analysis.png'")
        plt.show()
        
        return self
    
    def generate_decision_guide(self):
        """Generate a practical decision-making guide"""
        print("\n📋 PRACTICAL DECISION-MAKING GUIDE")
        print("=" * 50)
        
        print("""
🎯 UNDERSTANDING TARGET CLASSES:

CLASS 1 (High-Value Targets):
• Customers with higher potential value
• More likely to respond to premium offers
• Better candidates for loyalty programs
• Higher lifetime value potential

CLASS 0 (Standard Customers):
• Reliable but lower-value customers
• Good for basic service offerings
• Focus on retention and satisfaction
• Potential for upselling

📊 HOW TO USE CUTOFF VALUES:

1. **AUTOMATED SEGMENTATION**:
   - Use cutoffs to automatically classify new customers
   - Example: If income > $X AND spending_score > Y → Class 1

2. **TARGETED MARKETING**:
   - Class 1: Premium offers, exclusive services
   - Class 0: Standard offers, retention campaigns

3. **RESOURCE ALLOCATION**:
   - Allocate more resources to Class 1 segments
   - Optimize cost-to-serve for Class 0
        """)
        
        # Specific action plans
        print("\n🔍 ACTION PLANS BY SEGMENT:")
        
        if 'income' in self.cutoff_ranges and 'spending_score' in self.cutoff_ranges:
            income_cutoffs = self.cutoff_ranges['income']['cutoffs']
            spending_cutoffs = self.cutoff_ranges['spending_score']['cutoffs']
            
            if len(income_cutoffs) > 0 and len(spending_cutoffs) > 0:
                print(f"\n💎 PREMIUM SEGMENT (High Probability of Class 1):")
                print(f"   • Income > ${income_cutoffs[-1]:.0f}")
                print(f"   • Spending Score > {spending_cutoffs[-1]:.0f}")
                print(f"   🎯 Action: Personal account manager, premium offers")
                
                print(f"\n📈 GROWTH SEGMENT (Potential for Class 1):")
                print(f"   • Income ${income_cutoffs[0]:.0f}-${income_cutoffs[-1]:.0f}")
                print(f"   • Mixed spending behavior")
                print(f"   🎯 Action: Upsell campaigns, loyalty programs")
                
                print(f"\n🛡️  CORE SEGMENT (Typically Class 0):")
                print(f"   • Income < ${income_cutoffs[0]:.0f}")
                print(f"   • Conservative spending")
                print(f"   🎯 Action: Cost-effective service, basic offerings")
        
        return self
    
    def create_interactive_decision_tool(self):
        """Create a simple interactive decision tool"""
        print("\n🛠️  INTERACTIVE DECISION TOOL")
        print("=" * 40)
        print("Enter customer details to predict their segment:")
        
        try:
            # Get user input with defaults
            age = float(input("Age (default 35): ") or "35")
            income = float(input("Income (default 50000): ") or "50000")
            spending = float(input("Spending Score (default 50): ") or "50")
            credit = float(input("Credit Score (default 650): ") or "650")
            satisfaction = float(input("Satisfaction 1-5 (default 3.5): ") or "3.5")
            
            # Make prediction
            customer_data = [[age, income, spending, credit, satisfaction]]
            prediction = self.tree_model.predict(customer_data)[0]
            probability = self.tree_model.predict_proba(customer_data)[0]
            
            print(f"\n🎯 PREDICTION RESULTS:")
            class_name = "HIGH-VALUE TARGET (Class 1)" if prediction == 1 else "STANDARD CUSTOMER (Class 0)"
            print(f"   Predicted: {class_name}")
            print(f"   Confidence: {max(probability):.1%}")
            print(f"   Probability: Class 0: {probability[0]:.1%}, Class 1: {probability[1]:.1%}")
            
            # Explain why this prediction
            print(f"\n🔍 WHY THIS PREDICTION?:")
            self.explain_prediction(age, income, spending, credit, satisfaction)
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("Using default values for demonstration...")
            # Use defaults for demonstration
            customer_data = [[35, 50000, 50, 650, 3.5]]
            prediction = self.tree_model.predict(customer_data)[0]
            probability = self.tree_model.predict_proba(customer_data)[0]
            print(f"Demo Prediction: Class {prediction} ({max(probability):.1%} confidence)")
        
        return self
    
    def explain_prediction(self, age, income, spending, credit, satisfaction):
        """Explain why a particular prediction was made"""
        print("   The decision was based on these cutoff comparisons:")
        
        customer_values = {
            'age': age,
            'income': income, 
            'spending_score': spending,
            'credit_score': credit,
            'satisfaction': satisfaction
        }
        
        for feature, value in customer_values.items():
            if feature in self.cutoff_ranges and len(self.cutoff_ranges[feature]['cutoffs']) > 0:
                cutoffs = self.cutoff_ranges[feature]['cutoffs']
                relevant_cutoff = None
                
                for cutoff in cutoffs:
                    if value <= cutoff:
                        relevant_cutoff = cutoff
                        break
                
                if relevant_cutoff:
                    print(f"   • {feature}: {value:.1f} <= {relevant_cutoff:.1f}")
                else:
                    print(f"   • {feature}: {value:.1f} > {cutoffs[-1]:.1f}")
            else:
                print(f"   • {feature}: {value:.1f} (no specific cutoff in this tree)")

def main():
    print("🚀 COMPREHENSIVE DECISION CUTOFF ANALYSIS")
    print("=" * 60)
    
    try:
        analyzer = DecisionAnalysis()
        (analyzer.load_data()
                .train_and_analyze_tree(max_depth=4)
                .generate_decision_guide()
                .create_interactive_decision_tool())
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Generated Output:")
        print("   - decision_cutoff_analysis.png (Visualization)")
        print("   - Detailed target class analysis")
        print("   - Business decision rules")
        print("   - Interactive prediction tool")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()