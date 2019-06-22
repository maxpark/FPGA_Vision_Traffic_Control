def bin2float(bin_num,exponent_len = 5,mantissa_len = 11):
    """
    input:
        bin_num      - A string containing the binary patern
        exponent_len - Length of the exponent
    """
    mantissa = bin2int(bin_num[0:mantissa_len])
    exponent = bin2int(bin_num[mantissa_len:])
    if mantissa>0:
        suffix = ''
    else:
        suffix = '-'
    print (f'Mantissa = {mantissa}  Exponent = {exponent}')
    print (f"Number = {suffix}0.{abs(mantissa)} x 10^{exponent}")
    return (float(suffix+"0."+str(abs(mantissa)))*10**(exponent))


def bin2int(num):
    bin_num = num
    if bin_num[0]=='0' : # Positive
        return int(bin_num,2)
    else:
        bin_num = bin(int(bin_num,2)-1)
        bin_num = list(bin_num[2:])
        for i in range(len(bin_num)):
            if bin_num[i] == '0':
                bin_num[i] = '1'
            else:
                bin_num[i] = '0'
        bin_num = ''.join(bin_num)
        return int('-'+bin_num,2)

a = bin2float('1101100010111101')
print("Float  = ",a)

def float2bin(num):
    abs_num = abs(num)
    counter = 0
    