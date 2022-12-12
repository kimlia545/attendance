import os

def name_change(name):
    old_name = 'C:\\Users\\admin\\Desktop\\TMP\\knowns\\' +'user' + ".jpg" 
    new_name = "C:\\Users\\admin\\Desktop\\TMP\\knowns\\" + name  + ".jpg" 
    os.rename(old_name, new_name)


#name = 'Kim'    
#name_change(name)