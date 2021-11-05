import numpy as np

#função para ler as matrizes de testes
def ler_matriz(nome_arq):
    arq = open(nome_arq, 'r')
    
    n = int(arq.readline())                                    #lê a ordem da matriz
    A = np.zeros((n,n))                                        #inicializa matriz com ordem n e zeros
    
    i = 0
    while i < n: #enquanto não chegar ao final do arquivo
        linha = arq.readline()                                 #lê as linhas com os valores da matriz
        lista = np.array(linha.split())
        for ele in lista:
            ele = float(ele)                                   #faz o cast de string para float
        A[i] = lista
        i += 1
    
    arq.close()
    return n, A

#função que transforma elementos de uma lista de string para float
def lista_str_float(lista):
    L_aux = np.zeros(len(lista))
    for i in range(len(lista)):
        L_aux[i] = float(lista[i])                             #faz o cast de string para float
    return L_aux

#função que realiza as transformações de Householder
def householder(A):
    
    n = len(A[0])                                              #define n de acordo com a ordem da matriz A
    HT = np.eye(n)                                             #inicializa HT como matriz identidade de ordem n
    
    for i in range(n-2):
        
        x = np.zeros(n)
        x = np.array(A.T[i ,i+1:n])                            #vetor com os elementos da coluna que serão zerados na iteração i
        y = np.zeros(n-1-i)
        y[0] = sgn(x[0])                                       #vetor unitário com o sinal já estabelecido do 1° elemento de x 
        
        w_ = x + norma(x)*y                                    #define-se w_ transposto para a iterção i
        w = np.zeros(n)
        w[i+1:n] = w_                                          #define-se w transposto para a iteração i
        
        aux1 = np.zeros((n,n))
        aux1[0] = w
        w_wt = (aux1.T @ aux1)                                 #produto de w por w transposto
        H = np.eye(n) - (2/norma(w)**2)*w_wt                   #determinação da matriz Hw para o w da iteração i
        HT = HT @ H                                            #Ajuste de H transposta até a iterção i (completa após todas)
        A = H @ A @ H.T                                        #Ajuste de A até a iterção i (completa após todas)
        
    return A, HT                                               #retorna A e HT

#função que obtem os autval e autvet de matriz real simetrica
def get_auts(A):
    A_tri, HT = householder(A)                                 #obtem a matriz tridiagonalizada e HT
    autval_diagonal, autvet = QR_tri(A_tri, HT)                #obtem as matrizes com os auto-valores e auto-vetores
    autval = np.zeros(n)
    for i in range(n):
        autval[i] = autval_diagonal[i,i]                       #obtem um vetor com os autovalores
    
    return autval, autvet

#função para ler as matrizes de aplicação
def ler_trelicas(nome_arq):
    arq = open(nome_arq, 'r')
    
    linha0 = arq.readline()                                       #lê a primeira linha do arquivo
    lista0 = lista_str_float(np.array(linha0.split()))            #transforma os elementos de string para float
    n_nos = int(lista0[0])                                        #lê o número total de nós
    n_nos_moveis = int(lista0[1])                                 #lê o número de nós móveis
    n_barras = int(lista0[2])                                     #lê o numero de barras
    
    K = np.zeros((2*n_nos_moveis,2*n_nos_moveis))                 #inicializa a matriz de rigidez total com zeros
    
    linha1 = arq.readline()                                       #lê a segunda linha do arquivo
    lista1 = lista_str_float(np.array(linha1.split()))            #transforma os elementos de string para float
    ro = lista1[0]                                                #lê o valor da densidade
    a = lista1[1]                                                 #lê o valor da área da seção transversal
    E = lista1[2]*(10**9)                                         #lê o valor do módulo de elasticidade
    
    m = np.zeros(n_nos_moveis)

    for k in range(n_barras):                                     #laço para ler as demais linhas do arquivo
        linha = arq.readline()                                    #lê a linha k do laço (linha k+3 do arquivo)
        lista = lista_str_float(np.array(linha.split()))          #transforma os elementos de string para float
        i = int(lista[0])                                         #lê um dos nós extremos da barra {i,j}
        j = int(lista[1])                                         #lê o outro nó extremo da barra {i,j}
        theta = np.radians(lista[2])                              #lê o ângulo que a barra {i,j} faz com a horizontal
        l = lista[3]                                              #lê o comprimento da barra {i,j}
        c = np.cos(theta)                                         #calcula o cosseno do ângulo
        s = np.sin(theta)                                         #calcula o cosseno do ângulo
        
        #acréscimo de massa aos nós i e j
        if i <= n_nos_moveis:                                     #verifica se não se trata de um nó fixo
            m[i-1] += 0.5*ro*a*l                                  #acrescenta a contribuição de massa ao nó i
        if j <= n_nos_moveis:                                     #verifica se não se trata de um nó fixo       
            m[j-1] += 0.5*ro*a*l                                  #acrescenta a contribuição de massa ao nó j
        
        #calculo da matriz de rigidez da barra {i,j}
        Kij = (a*E/l) * np.array([[c**2, c*s, -c**2, -c*s],              
                                 [c*s, s**2, -c*s, -s**2],               
                                 [-c**2, -c*s, c**2, c*s],               
                                 [-c*s, -s**2, c*s, s**2]])   
        
        #calculo da matriz de rigidez total
        pos_2i_1 = (2*i-1)-1
        pos_2i = 2*i-1
        pos_2j_1 = (2*j-1)-1
        pos_2j = 2*j-1
        
        K[pos_2i_1, pos_2i_1] += Kij[0,0]
        K[pos_2i_1, pos_2i] += Kij[0,1]
        
        K[pos_2i, pos_2i_1] += Kij[1,0]
        K[pos_2i, pos_2i] += Kij[1,1]
        
        if j <= n_nos_moveis:                                       #não altera para os nós fixos
            K[pos_2i, pos_2j_1] += Kij[1,2]
            K[pos_2i, pos_2j] += Kij[1,3]
            
            K[pos_2i_1, pos_2j_1] += Kij[0,2]
            K[pos_2i_1, pos_2j] += Kij[0,3]

            K[pos_2j_1, pos_2i_1] += Kij[2,0]
            K[pos_2j_1, pos_2i] += Kij[2,1]
            K[pos_2j_1, pos_2j_1] += Kij[2,2]
            K[pos_2j_1, pos_2j] += Kij[2,3]

            K[pos_2j, pos_2i_1] += Kij[3,0]
            K[pos_2j, pos_2i] += Kij[3,1]
            K[pos_2j, pos_2j_1] += Kij[3,2]
            K[pos_2j, pos_2j] += Kij[3,3]
            
    arq.close()
    return m, K


#função que cria a matriz M
def cria_M(m):
    M = np.zeros((2*len(m),2*len(m)))
    for i in range(len(m)):
        M[2*i,2*i] = m[i]                                      #adiciona mi em uma posição da diagonal
        M[2*i+1, 2*i+1] = m[i]                                 #adiciona mi na próxima posição da diagonal
    return M


#função que cria a matriz M^(-1/2)
def cria_M_sqrt(M):
    M_sqrt = np.zeros((len(M),len(M)))
    for i in range(len(M)):
        M_sqrt[i,i] = 1/(np.sqrt(M[i,i]))                      #calcula o inverso da raiz das massas
    return M_sqrt


#função que gera K_tio
def get_K_tio(K, M):
    M_sqrt = cria_M_sqrt(M)                                    #obtém M^(-1/2)
    K_tio = np.zeros((len(K),len(K)))                          
    K_tio = M_sqrt @ K @ M_sqrt
    return K_tio


#função que gera as frequências de vibração 
def get_w(autval):
    w = np.zeros(len(autval))
    for i in range (len(autval)):
        w[i] = np.sqrt(autval[i])                              #raiz quadrada de cada um dos elmentos
    return w


#função que encontra as 5 menores frequências de vibração
def get_w_5(w):
    w_5 = np.zeros(5)
    w_aux = np.sort(w)                                       #organiza os elemtnos de forma crescente
    for i in range(5):
        w_5[i] = w_aux[i]                                    #obtém as 5 menores frequências
    return w_5

#função que encontra os 5 modos de vibração de menor enrgia
def get_modos_5(modos, w, w_5):
    modos_5 = np.zeros((len(w_5), len(modos[0])))
    for i in range(len(w_5)):
        pos = np.where(w==w_5[i])                       #encontra o índice das 5 menores frequências 
        modos_5[i] = modos.T[pos]                       #a partir dos índices das frequências, toma-se seus os respectivos modos
    return modos_5.T


#função para calcular a norma de um vetor
def norma(x):
    k = np.zeros(len(x))
    k = x*x.T                                                  #faz o produto dos elementos de x com eles próprios
    return np.sqrt(sum(k))                                     #retorna a soma dos elementos de k


#função sgn()
def sgn(d):    
    if d >= 0:
        return 1                                               #d >= 0
    else:
        return -1                                              #d < 0

#função que gera um valor de d_k
def d_gen(A, m):      
    alpha_m1 = A[m-2,m-2]                                      #penultimo elemento do conjunto alpha
    alpha_m = A[m-1,m-1]                                       #ultimo elemento do conjunto alpha
    return (alpha_m1 - alpha_m)/2                              #calculo de d_k


#função que gera um valor de mi para cada iteração k
def mi_gen(A, m):     
    d = d_gen(A, m)                                            #geração de d_k                  
    alpha_m = A[m-1,m-1]                                       #ultimo elemento do conjunto alpha
    beta_m1 = A[m-1,m-2]                                       #penultimo elemento do conjunto beta
    return alpha_m + d - sgn(d)*np.sqrt(d**2 + beta_m1**2)     #calculo de mi_k
 

#função que gera c e s para cada Q de cada iteração k
def cs_gen(alpha, beta):     
    if abs(alpha) > abs(beta):                                 #testa se |alpha| > |beta|
        tau = -beta/alpha                                      #calculo de tau
        c = 1/(np.sqrt(1+tau**2))                              #calculo de c
        s = c*tau                                              #calculo de s
    else:
        tau = -alpha/beta                                      #calculo de tau
        s = 1/(np.sqrt(1+tau**2))                              #calculo de s
        c = s*tau                                              #calculo de c
    return [c, s]


#função que gera Q para cada iteração k 
def Q_gen(i, c, s):      
    Q = np.eye(n)                                              #inicia Q como matriz identidade de ordem n
    Q[i,i] = c                                                 #substitui o elemento da posição (i,i) por c
    Q[i+1,i] = s                                               #substitui o elemento da posição (i+1,i) por s
    Q[i,i+1] = -s                                              #substitui o elemento da posição (i,i+1) por -s
    Q[i+1,i+1] = c                                             #substitui o elemento da posição (i+1,i+1) por c
    return Q


#função que obtem os autval e autvet de matriz tridiagonal simetrica
def QR_tri(A, HT):
    V = HT                                                     #inicializa a matriz de auto-vetores com HT
    n = len(A[0])
    ck = np.zeros(n-1)                                         #vetor que armazena os valores dos cossenos em cada iteração
    sk = np.zeros(n-1)                                         #vetor que armazena os valores dos senos em cada iteração
    erro = 10**(-6)                                            #erro para o critério de parada
    
    #Algoritmo QR tridiagonal com deslocamento
    k = 0                                                      #parâmetro que representa a quantidade de iterações realizadas
    m = n                                                      #auxiliar que segmenta A sempre que atinge uma condição de parada
    Ak = A                                                     #matriz auxiliar que muda a cada iteração

    while m > 1:                                               #m vai de n até 2
        fim = False
        while fim == False:
            mi = 0

            if k > 0:                                          #testa se k > 0
                mi = mi_gen(Ak, m)                             #gera mi_k
            Ak =  Ak - mi * np.eye(n)                          #subtrai deslocamento espectral

            for i in range(m-1):
                cs = cs_gen(Ak[i,i], Ak[i+1, i])               #gera c e s
                ck[i] = cs[0]                                  #armazena os valores de c da iteração k
                sk[i] = cs[1]                                  #armazena os valores de s da iteração k
                Q = Q_gen(i, ck[i], sk[i])                     #gera a matriz Q(i,i+1,theta), onde c=cos(theta) e s=sen(theta)
                Ak = Q @ Ak                                    #realiza as rotações de Givens

            for i in range(m-1):
                Q = Q_gen(i, ck[i], sk[i])                     #gera a matriz Q(i,i+1,theta), onde c=cos(theta) e s=sen(theta)
                Ak = Ak @ Q.T                                  #atualiza a matriz Ak

            Ak = Ak + mi * np.eye(n)                           #soma deslocamento espectral

            for i in range(m-1):
                Q = Q_gen(i, ck[i], sk[i])                     #gera a matriz Q(i,i+1,theta), onde c=cos(theta) e s=sen(theta)
                V = V @ Q.T                                    #atualiza a matriz de auto-vetores

            k += 1                                             #atualiza a quantidade de iterações

            beta_m1 = Ak[m-2, m-1]                             #elemento que deve ser menor que o erro
            if abs(beta_m1) < erro:                            #condição de parada
                fim = True                                       

        m -= 1                                                 #decrementa 1 de m

    return Ak, V                                               #armazena os auto-valores na matriz AUTVAL


#interface com o usuário
tarefa = str(input("Qual parte do exercício-progrma deseja executar? (t [para testes] ou a [aplicação para as treliças] "))

if tarefa == 't':
    teste = str(input("Qual teste deseja ser utilizado? (a ou b) "))
    
    if teste == 'a':
        n, A = ler_matriz('input-a')
        
    elif teste == 'b':
        n, A = ler_matriz('input-b')
        
    A_ini = A
    
if tarefa == 'a':
    m, K = ler_trelicas('input-c')
    n = len(K)
    
    
if tarefa == 't':
    autval, autvet = get_auts(A)                                   #determinação dos autovalores e autovetores de A inicial
    print('-------------------------------------------------')
    print("Matriz A inicial: \n")
    print(A_ini)
    print('')
    print("Autovalores de A: \n")
    print(autval)
    print('')
    print("Autovetores de A: \n")
    print(autvet)
    print('')
    print('-------------------------------------------------')
    for i in range(n):
        Av = A_ini @ autvet.T[i]                                   #produto matricial de A inicial por um de seus autovetores
        lambdav = autval[i] * autvet.T[i]                          #produto do autovalor pelo seu autovetor em questão 
        print('-------------------------------------------------')
        print("Produto da matriz A pelo autovetor %s: \n" % (i+1))
        print(Av)
        print('')
        print("Produto do autovalor %s pelo autovetor %s: \n" % (i+1, i+1))
        print(lambdav)
        print('')
        print('-------------------------------------------------')
    print('-------------------------------------------------')
    print("Verificação da ortogonalidade da matriz de autovetores (basta que a transposta seja igual a inversa) \n")
    print('Matriz de autovetores transposta: \n')
    print(autvet.T)
    print('')
    print('Matriz de autovetores invertida: \n')
    print(np.linalg.inv(autvet))
    print('-------------------------------------------------')

if tarefa == 'a':
    M = cria_M(m)                                                  #criação da matriz diagonal M de massas como epecificada
    K_tio = get_K_tio(K, M)                                        #criação de K~ 
    autval, autvet = get_auts(K_tio)                               #obtenção dos autovalores e autovetores de K~
    w = get_w(autval)                                              #obtenção das frequências de vibração a partir dos autovalores
    w_5 = get_w_5(w)                                               #obtenção das 5 menores frequências
    modos = cria_M_sqrt(M) @ autvet                                #obtenção dos modos a partir dos autovetores de K~
    modos_5 = get_modos_5(modos, w, w_5)                           #obtenção dos 5 modos de vibração de menor energia
    print('-------------------------------------------------')
    print("As 5 menores frequências de virbração do sistema: \n")
    print(w_5)
    print('')
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    print("Os 5 modos de vibração de menor energia do sistema: \n")
    print(modos_5)
    print('-------------------------------------------------')





