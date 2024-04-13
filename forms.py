from django import forms
from .models import ckdModel



class ckdForm(forms.ModelForm):

    class Meta():
        model=ckdModel
        fields=['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','JobSatisfaction']
