import random
from bso_sumo import main as BSOmain
from GA_sumo import main as GAmain
import os

root_dir = os.path.dirname(__file__)

anaheim_net = os.path.join(root_dir, "output", "anaheim.net.xml")
anaheim_cfg = os.path.join(root_dir, "output", "anaheim.sumocfg")
print(anaheim_cfg)
def cust_str(n):
    customers_int = random.sample(range(1, 417), n)
    customers_str = [ str(customer) for customer in customers_int]
    return  customers_str
#GAmain(anaheim_cfg, anaheim_net, str(random.randint(1,417)), cust_str(7), 2)
BSOmain(anaheim_cfg, anaheim_net, str(random.randint(1,417)), cust_str(7), 2)
