# Generated by Django 2.2 on 2021-06-24 03:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='expand_sortlist',
            name='left_fragment',
        ),
        migrations.RemoveField(
            model_name='expand_sortlist',
            name='right_fragment',
        ),
        migrations.RemoveField(
            model_name='expand_sortlist',
            name='transformation',
        ),
        migrations.RemoveField(
            model_name='expand_sortlist',
            name='transformation_reaction_SMARTS',
        ),
    ]