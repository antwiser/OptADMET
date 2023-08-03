from django.db import models
from django.db.models.fields import DateTimeField
from django.db.models.fields.related import ManyToManyField


class ADMETProperty(models.Model):
    _id = models.AutoField(primary_key=True)
    md5 = models.CharField(default='', max_length=128)
    SMILES = models.CharField(max_length=1024)
    LogS = models.FloatField()
    LogD = models.FloatField()
    LogP = models.FloatField()
    pgpinh = models.FloatField()
    pgpsub = models.FloatField()
    hia = models.FloatField()
    f20 = models.FloatField()
    f30 = models.FloatField()
    caco2 = models.FloatField()
    mdck = models.FloatField()
    bbb = models.FloatField()
    ppb = models.FloatField()
    vdss = models.FloatField()
    fu = models.FloatField()
    cyp1a2inh = models.FloatField()
    cyp1a2sub = models.FloatField()
    cyp2c19inh = models.FloatField()
    cyp2c19sub = models.FloatField()
    cyp2c9inh = models.FloatField()
    cyp2c9sub = models.FloatField()
    cyp2d6inh = models.FloatField()
    cyp2d6sub = models.FloatField()
    cyp3a4inh = models.FloatField()
    cyp3a4sub = models.FloatField()
    cl = models.FloatField()
    t12 = models.FloatField()
    herg = models.FloatField()
    hht = models.FloatField()
    dili = models.FloatField()
    ames = models.FloatField()
    roa = models.FloatField()
    fdamdd = models.FloatField()
    skinsen = models.FloatField()
    carcinogenicity = models.FloatField()
    ec = models.FloatField()
    ei = models.FloatField()
    respiratory = models.FloatField()
    bcf = models.FloatField()
    igc50 = models.FloatField()
    lc50 = models.FloatField()
    lc50dm = models.FloatField()
    nr_ar = models.FloatField()
    nr_ar_lbd = models.FloatField()
    nr_ahr = models.FloatField()
    nr_aromatase = models.FloatField()
    nr_er = models.FloatField()
    nr_er_lbd = models.FloatField()
    nr_ppar_gamma = models.FloatField()
    sr_are = models.FloatField()
    sr_atad5 = models.FloatField()
    sr_hse = models.FloatField()
    sr_mmp = models.FloatField()
    sr_p53 = models.FloatField()

    def to_dict(self, fields=None, exclude=None):
        data = {}
        for f in self._meta.concrete_fields + self._meta.many_to_many:
            value = f.value_from_object(self)

            if fields and f.name not in fields:
                continue

            if exclude and f.name in exclude:
                continue

            if isinstance(f, ManyToManyField):
                value = [i.id for i in value] if self.pk else None

            if isinstance(f, DateTimeField):
                value = value.strftime("%Y-%m-%d %H:%M:%S") if value else None

            data[f.name] = value

        return data

    class Meta:
        verbose_name = "ADMET Information"
        verbose_name_plural = "ADMET Information"
