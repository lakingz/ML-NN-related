class Employee:

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @fullname.deleter
    def fullname(self):
        print('Delete name:' + self.fullname)
        self.first = None
        self.last = None

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    def __repr__(self):
        return "Employee（‘{}'， '{}', {})".format(self.first, self.last, self.pay)

    def __str__(self):
        return '{} - {}'.format(self.fullname(), self.raise_amt)

    def printraise(self):
        print(Employee.raise_amt)
    
class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, program_lang):
        super().__init__(first, last, pay)
        self.program_lang = program_lang

class Manager(Employee):

    def __init__(self, first, last, pay, employees = None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees is None
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())
    

    
mgr_1 = Manager('sue', 'smith', 90000, [dev_1])
print(mgr_1.email)
mgr_1.add_emp(emp_2)
mgr_1.remove_emp(emp_2)
print(mgr_1.print_emps())

print(repr(emp_1))
print(str(emp_1))

print(help(Employee))

emp_1 = Employee('cor', '12', 50000)
emp_1.first = 'jax'
print(emp_1.email)
print(emp_1.fullname)
emp_1.fullname = 'jaw 4'
del emp_1.fullname

emp_2 = Employee('test', 'user', 60000)
print(emp_1 + emp_2)

dev_1 = Developer('tdev', 'user', 60000, 'python')

print(dev_1.pay)
dev_3.apply_raise()
print(dev_3.pay)
print(Employee.__dict__)
print(emp_1.__str__)

a = 'test'
def rec(a):
    if len(a) == 4:
        return('True')
    else:
        return('else')
leng(a)
print(emp_1.fullname)
Employee.rec(emp_1)

def rec():
    for i in range(10):
        if i > 8:
            return i
rec()
for i in range(10):
    print(i)

emp_1.printraise()
emp_2.raise_amt = 100
Employee.printraise(emp_2)