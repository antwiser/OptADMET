from django.db import models


class Experi_Sortlist(models.Model):
    sortlist1 = models.CharField(max_length=512, blank=True, null=True)
    sortlist2 = models.CharField(max_length=512, blank=True, null=True)
    sortlist = models.CharField(max_length=512, blank=True, null=True)
    structure_global_id = models.CharField(max_length=16, primary_key=True)
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)

    def __str__(self):
        return self.structure_global_id

    class Meta:
        verbose_name = "Structure Information"
        verbose_name_plural = "Structure Information"


class Expand_Sortlist(models.Model):
    sortlist1 = models.CharField(max_length=512, blank=True, null=True)
    sortlist2 = models.CharField(max_length=512, blank=True, null=True)
    sortlist = models.CharField(max_length=512, blank=True, null=True)
    structure_global_id = models.CharField(max_length=16, primary_key=True)
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)

    # transformation = models.CharField(max_length=256, blank=True, null=True)
    # left_fragment = models.CharField(max_length=128, blank=True, null=True)
    # right_fragment = models.CharField(max_length=128, blank=True, null=True)
    # transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)

    def __str__(self):
        return self.structure_global_id

    class Meta:
        verbose_name = "Structure Information"
        verbose_name_plural = "Structure Information"


class Experi_global(models.Model):
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)
    average_change = models.FloatField(blank=True, null=True)
    count = models.FloatField(blank=True, null=True)
    nochange_portion = models.FloatField(blank=True, null=True)
    increase_portion = models.FloatField(blank=True, null=True)
    decrease_portion = models.FloatField(blank=True, null=True)
    transformation_global_ID = models.CharField(max_length=64, primary_key=True)
    statistical_significance = models.FloatField(blank=True, null=True)
    variance = models.FloatField(blank=True, null=True)
    structure_global_ID = models.ForeignKey(Experi_Sortlist, on_delete=models.CASCADE)

    def __str__(self):
        return self.transformation_global_ID

    class Meta:
        ordering = ['statistical_significance', 'variance']
        verbose_name = "Global Information"
        verbose_name_plural = "Global Information"


class Expand_global(models.Model):
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)
    average_change = models.FloatField(blank=True, null=True)
    count = models.FloatField(blank=True, null=True)
    nochange_portion = models.FloatField(blank=True, null=True)
    increase_portion = models.FloatField(blank=True, null=True)
    decrease_portion = models.FloatField(blank=True, null=True)
    transformation_global_ID = models.CharField(max_length=64, primary_key=True)
    statistical_significance = models.FloatField(blank=True, null=True)
    variance = models.FloatField(blank=True, null=True)
    structure_global_ID = models.ForeignKey(Expand_Sortlist, on_delete=models.CASCADE)

    def __str__(self):
        return self.transformation_global_ID

    class Meta:
        ordering = ['statistical_significance', 'variance']
        verbose_name = "Global Information"
        verbose_name_plural = "Global Information"


class Experi_local(models.Model):
    transformation_global_ID = models.ForeignKey(Experi_global, on_delete=models.CASCADE)
    structure_global_ID = models.ForeignKey(Experi_Sortlist, on_delete=models.CASCADE)
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)
    env_1 = models.CharField(max_length=32, blank=True, null=True)
    env_2 = models.CharField(max_length=32, blank=True, null=True)
    env_3 = models.CharField(max_length=32, blank=True, null=True)
    average_change = models.FloatField(blank=True, null=True)
    count = models.FloatField(blank=True, null=True)
    nochange_portion = models.FloatField(blank=True, null=True)
    increase_portion = models.FloatField(blank=True, null=True)
    decrease_portion = models.FloatField(blank=True, null=True)
    transformation_local_ID = models.CharField(max_length=64, primary_key=True)
    statistical_significance = models.FloatField(blank=True, null=True)
    variance = models.FloatField(blank=True, null=True)
    structure_local_ID = models.CharField(max_length=64, blank=True, null=True)

    def __str__(self):
        return self.transformation_local_ID

    class Meta:
        verbose_name = "Local Information"
        verbose_name_plural = "Local Information"


class Expand_local(models.Model):
    transformation_global_ID = models.ForeignKey(Expand_global, on_delete=models.CASCADE)
    structure_global_ID = models.ForeignKey(Expand_Sortlist, on_delete=models.CASCADE)
    transformation = models.CharField(max_length=256, blank=True, null=True)
    left_fragment = models.CharField(max_length=128, blank=True, null=True)
    right_fragment = models.CharField(max_length=128, blank=True, null=True)
    transformation_reaction_SMARTS = models.CharField(max_length=256, blank=True, null=True)
    env_1 = models.CharField(max_length=32, blank=True, null=True)
    env_2 = models.CharField(max_length=32, blank=True, null=True)
    env_3 = models.CharField(max_length=32, blank=True, null=True)
    average_change = models.FloatField(blank=True, null=True)
    count = models.FloatField(blank=True, null=True)
    nochange_portion = models.FloatField(blank=True, null=True)
    increase_portion = models.FloatField(blank=True, null=True)
    decrease_portion = models.FloatField(blank=True, null=True)
    transformation_local_ID = models.CharField(max_length=64, primary_key=True)
    statistical_significance = models.FloatField(blank=True, null=True)
    variance = models.FloatField(blank=True, null=True)
    structure_local_ID = models.CharField(max_length=64, blank=True, null=True)

    def __str__(self):
        return self.transformation_local_ID

    class Meta:
        verbose_name = "Local Information"
        verbose_name_plural = "Local Information"


class Experi_MMP(models.Model):
    change = models.FloatField(blank=True, null=True)
    value_l = models.FloatField(blank=True, null=True)
    value_r = models.FloatField(blank=True, null=True)
    molecule_l = models.CharField(max_length=512, blank=True, null=True)
    molecule_r = models.CharField(max_length=512, blank=True, null=True)
    transformation_global_ID = models.ForeignKey(Experi_global, on_delete=models.CASCADE)
    transformation_local_ID = models.ForeignKey(Experi_local, on_delete=models.CASCADE)

    def __str__(self):
        return self.transformation_global_ID.transformation_global_ID + self.transformation_local_ID.transformation_local_ID

    class Meta:
        verbose_name = "MMP Information"
        verbose_name_plural = "MMP Information"


class Expand_MMP(models.Model):
    change = models.FloatField(blank=True, null=True)
    value_l = models.FloatField(blank=True, null=True)
    value_r = models.FloatField(blank=True, null=True)
    molecule_l = models.CharField(max_length=512, blank=True, null=True)
    molecule_r = models.CharField(max_length=512, blank=True, null=True)
    transformation_global_ID = models.ForeignKey(Expand_global, on_delete=models.CASCADE)
    transformation_local_ID = models.ForeignKey(Expand_local, on_delete=models.CASCADE)

    def __str__(self):
        return self.transformation_global_ID.transformation_global_ID + self.transformation_local_ID.transformation_local_ID

    class Meta:
        verbose_name = "MMP Information"
        verbose_name_plural = "MMP Information"


class Property(models.Model):
    property = models.CharField(max_length=32, blank=True, null=True)
    idx = models.IntegerField(primary_key=True)

    def __str__(self):
        return self.property

    class Meta:
        verbose_name = "Property Information"
        verbose_name_plural = "Property Information"


class Experi_Property_Structure(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    structure = models.ForeignKey(Experi_Sortlist, on_delete=models.CASCADE)

    def __str__(self):
        return self.property.property + '_' + self.structure.structure_global_id

    class Meta:
        verbose_name = "Property Information"
        verbose_name_plural = "Property Information"


class Expand_Property_Structure(models.Model):
    property = models.ForeignKey(Property, on_delete=models.CASCADE)
    structure = models.ForeignKey(Expand_Sortlist, on_delete=models.CASCADE)

    def __str__(self):
        return self.property.property + '_' + self.structure.structure_global_id

    class Meta:
        verbose_name = "Property Information"
        verbose_name_plural = "Property Information"
