from django.db import models

# Create your models here.
class ckdModel(models.Model):

    YearsAtCompany=models.FloatField()
    YearsInCurrentRole=models.FloatField()
    YearsSinceLastPromotion=models.FloatField()
    YearsWithCurrManager=models.FloatField()
    JobSatisfaction=models.FloatField()
    MonthlyIncome=models.FloatField()
