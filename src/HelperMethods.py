def create_salaried_age_3(occ,age):
    if (occ == "Salaried" and str(age)=="3"):
        return 1
    else:
        return 0

def create_credit_product_missing(cp):
    if cp == "None":
        return 1
    else: 
        return 0

def create_salaried_credit_product_missing(occ,cp):
    if occ=="Salaried" and cp == "None":
        return 1
    else: 
        return 0

def create_active_age_1(act,age):
    if act=="Yes" and str(age)=="1":
        return 1
    else:
        return 0

def create_active_entrepreneur(act,occ):
    if act=="Yes" and occ=="Entrepreneur":
        return 1
    else:
        return 0

def create_active_salaried(act,occ):
    if act=="Yes" and occ=="Salaried":
        return 1
    else:
        return 0

def create_active_other(act,occ):
    if act=="Yes" and occ=="Other":
        return 1
    else:
        return 0

def create_active_self_employed(act,occ):
    if act=="Yes" and occ=="Self_Employed":
        return 1
    else:
        return 0

def create_salaried_x2(cc,occ):
    if cc=="X2" and occ=="Salaried":
        return 1
    else:
        return 0

def create_salaried_x3(cc,occ):
    if cc=="X3" and occ=="Salaried":
        return 1
    else:
        return 0