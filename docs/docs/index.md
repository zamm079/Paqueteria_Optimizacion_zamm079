# Paquete de metodos de optimizacion

Repositorio [repositoriGitHub](https://github.com/zamm079/Paquete_optimizacion_byzamm079)

Libreria [Lib PyPi](https://pypi.org/project/Paquete-optimizacion-byzamm079/)

Este documento cuentas con distintas funciones de optimizacion, esta consiste en encontrar la mejor solución posible, cambiando el valor de las variables que pueden ser controladas, algunas veces sujeto a restricciones.
La optimización tiene un amplio atractivo porque es aplicable en todos los dominios y debido al deseo humano de mejorar las cosas. Cualquier problema en el que se deba tomar una decisión puede plantearse como un problema de optimización.
 

## Funciones

### Metodos univariable:

#### Métodos de eliminación de regiones

* `interval_halving(a,b,e,funcion)` Método de división de intervalos por la mitad

        def interval_halving(a,b,e,funcion):
        #step 1
        Xm=(a+b)/2
        L=b-a

        while abs(L) > e:
            # step 2
            x1 = a + L /4 #a
            x2 = b - L /4 #b
            fx1 = funcion(x1)
            fx2 = funcion(x2)
            Fxm = funcion(Xm)
            #step 3
            if fx1 < Fxm:
                b = Xm
                Xm = x1
            else:
                if fx2 < Fxm:
                    a = Xm
                    Xm=x2
                else:
                    a = x1
                    b = x2
            
            L = b-a
            if abs(L) < e:
                return [x1,x2]

* `fibonacci_search(a,b,n,funcion)` Búsqueda de Fibonacci

        def fibonacci_search(a,b,n,funcion):
            #step 1
            L = b-a
            k=2
            while k != n:
                #step 2
                Lk = fibonacci(n-k+1)/fibonacci(n+1)
                x1 = a + Lk
                x2 = b - Lk
                #step 3
                fx1 = funcion(x1)
                fx2 = funcion(x2)
                if fx1 > fx2:
                    a = x1
                if fx1 < fx2:
                    b = x2
                if fx1 == fx2:
                    a = x1
                    b = x2
                #step 4
                k = k+1
            return [x1,x2]

* `golden_section_search(funcion, a, b, epsilon)` Método de la sección dorada

        def golden_section_search(funcion, a, b, epsilon):
            
            golden = 0.618
            golden2 = 1 - golden

            w1 = a + golden2 * (b - a)
            w2 = a + golden * (b - a)

            f_x1 = funcion(w1)
            f_x2 = funcion(w2)

            while b - a > epsilon:
                
                if f_x1 < f_x2:
                    b = w2
                    w2 = w1
                    w1 = a + golden2 * (b - a)
                    f_x2 = f_x1
                    f_x1 = funcion(w1)
                else:
                    a = w1
                    w1 = w2
                    w2 = a + golden * (b - a)
                    f_x1 = f_x2
                    f_x2 = funcion(w2)

            return [a, b]

#### Métodos basados en la derivada

* `newton_raphson_method(funcion, i_guess, delta_fun, epsilon)` Método de Newton-Raphson

        def newton_raphson_method(funcion, i_guess, delta_fun, epsilon):
            x = i_guess
            k = 1
            max_iter=10000
            while k < max_iter:
                #step1
                delta_x = delta_fun(x)
                f_derivada1 = central_difference_1(funcion, x, delta_x)
                #step2
                f_derivada2= central_difference_2(funcion, x, delta_x)
                
                if abs(f_derivada1) < epsilon:
                    return x
                #step 3
                x_k1 = x - f_derivada1 / f_derivada2
                #step 4
                if abs(x_k1 - x) < epsilon:
                    return x_k1
                
                x = x_k1
                k += 1
            
            return x

* `bisection_method(funcion, a, b, epsilon, delta_x)` Método de bisección

        def bisection_method(funcion, a, b, epsilon, delta_x):
            x1 = a
            x2 = b
            max_iter=10000
            if (central_difference_1(funcion, a, delta_x) < 0) and (central_difference_1(funcion, b, delta_x) > 0):
                epsilon = epsilon
            else:
                raise ValueError("La función no cumple con la condición")
            
            iteraciones = 0

            while abs(x1 - x2) > epsilon and iteraciones < max_iter:
                z = (x1 + x2) / 2
                f_z = central_difference_1(funcion, z, delta_x)

                if abs(f_z) <= epsilon:
                    return z, z 

                if f_z < 0:
                    x1 = z
                else:
                    x2 = z

                iteraciones += 1

            return [x1, x2]

* `secant_method(funcion, a, b, epsilon, delta_x)` Método de la secante

        def secant_method(funcion, a, b, epsilon, delta_x):
            x1 = a
            x2 = b
            max_iter=10000
            if (central_difference_1(funcion, a, delta_x) < 0) and (central_difference_1(funcion, b, delta_x) > 0):
                epsilon = epsilon
            else:
                raise ValueError("La función no cumple con la condición")
            
            i = 0

            while abs(x1 - x2) > epsilon and i < max_iter:
                z = x2 - (central_difference_1(funcion, x2, delta_x)/((central_difference_1(funcion, x2, delta_x)-central_difference_1(funcion, x1, delta_x))/(x2-x1)))
                f_z = central_difference_1(funcion, z, delta_x)

                if abs(f_z) <= epsilon:
                    return z, z 

                if f_z < 0:
                    x1 = z
                else:
                    x2 = z

                i += 1

            return [x1, x2] 


### Metodos multivarible:

#### Métodos directos

* `met_random_walk(funcion,x0,epsilon,max_iter)` Caminata aleatoria

        def met_random_walk(funcion,x0,epsilon,max_iter):

            def gen_aleatorio(xk):
                return xk + np.random.uniform(-epsilon,epsilon,size=xk.shape)

            x_mejor = x0
            xk = x0
            iteraciones = 0
            while iteraciones < max_iter:
                xk1 = gen_aleatorio(xk)
                if funcion(xk1) < funcion(x_mejor):
                    x_mejor = xk1
                xk = xk1
                iteraciones += 1

            return x_mejor

* `simplex_search_meth(x,func)` `nelder_mead(func, x_start)` Método de Nelder y Mead (Simplex)

        def simplex_search_meth(x,func,gama=2.0,beta=0.2,epsilon=0.001):
            # step 1
            #no cero hipervolumen
            alpha=1
            N = len(x)
            d1 = ((math.sqrt(N+1)+N-1)/N*math.sqrt(2))*alpha
            d2 = ((math.sqrt(N+1)-1)/N*math.sqrt(2))*alpha
            simplex = np.zeros((N + 1,N))
            for i in range(len(simplex)):
                for j in range(N):
                    if j == i:
                        simplex[i,j] = x[j]+d1
                    if j != i:
                        simplex[i,j] = x[j]+d2
            i_max = 10
            i = 0

            # step 2
            f_values = np.apply_along_axis(func, 1, simplex)
            xi=0
            
            while i < i_max:
                val_orden = np.argsort(f_values)
                simplex = simplex[val_orden]
                xl,xg,xh = f_values[val_orden]
                #Xc
                xc = np.mean(simplex[:-1])
                i+=1
                #step 3
                xr = 2*xc - xh
                xnew = xr
                
                if func(xr) < func(xl):
                    xnew = (1+gama)*xc - (gama*xh) 
                elif func(xr) >= func(xh):
                    xnew = (1-beta)*xc+(beta*xh)
                elif func(xg) < func(xr) < func(xh):
                    xnew = (1+beta)*xc-(beta*xh)
                xh = xnew
                #step 4
                xi= np.sum(func(simplex))
                term1=np.sum((xi-xc)**2/(N+1))
                if term1**0.5 < epsilon:
                    break
            return xnew

        def nelder_mead(func, x_start, tol=1e-6, max_iter=1000):
            # Parámetros del algoritmo
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5


            n = len(x_start)
            simplex = np.zeros((n + 1, n))
            simplex[0] = x_start
            for i in range(n):
                y = np.array(x_start, copy=True)
                y[i] += 0.05 if x_start[i] == 0 else 0.05 * x_start[i]
                simplex[i + 1] = y


            f_values = np.apply_along_axis(func, 1, simplex)
            iter_count = 0
            
            while iter_count < max_iter:
                # Ordenar el simplex por los valores de la función
                indices = np.argsort(f_values)
                simplex = simplex[indices]
                f_values = f_values[indices]

                # Centroid de los mejores n puntos
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflejar
                xr = centroid + alpha * (centroid - simplex[-1])
                fxr = func(xr)

                if fxr < f_values[0]:

                    xe = centroid + gamma * (xr - centroid)
                    fxe = func(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        f_values[-1] = fxe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fxr
                else:
                    if fxr < f_values[-2]:
                        simplex[-1] = xr
                        f_values[-1] = fxr
                    else:
                        # Contracción
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fxc = func(xc)
                        if fxc < f_values[-1]:
                            simplex[-1] = xc
                            f_values[-1] = fxc
                        else:
                            # Reducción
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values = np.apply_along_axis(func, 1, simplex)
                
                iter_count += 1


                if np.max(np.abs(simplex[0] - simplex[1:])) < tol:
                    break
            
            return f_values[0]

* `hooke_jeeves(func, x0)` Método de Hooke-Jeeves

        def hooke_jeeves(func, x0, step_size=0.5, step_reduction=0.5, tolerance=1e-6, max_iterations=1000):
            n = len(x0)
            x = np.array(x0)
            best = np.copy(x)
            step = np.full(n, step_size)

            def explore(base_point, step_size):
                new_point = np.copy(base_point)
                for i in range(n):
                    for direction in [1, -1]:
                        candidate = np.copy(new_point)
                        candidate[i] += direction * step_size[i]
                        if func(candidate) < func(new_point):
                            new_point = candidate
                            break
                return new_point

            iteration = 0
            while np.max(step) > tolerance and iteration < max_iterations:
                new_point = explore(x, step)
                if func(new_point) < func(x):
                    best = new_point + (new_point - x)
                    x = new_point
                else:
                    step = step * step_reduction
                iteration += 1
                # print(f"Iteration {iteration}, x: {x}, f(x): {func(x)}")

            return x

#### Métodos de gradiente

* `cauchy(funcion,x0,epsilon1,epsilon2,M)` Método de Cauchy

        def cauchy(funcion,x0,epsilon1,epsilon2,M):

            terminar=False
            xk=x0
            k=0
            while not terminar:
                grad = np.array(gradiente(funcion,xk))

                if np.linalg.norm(grad) < epsilon1 or k >= M:
                    terminar=True
                else:

                    def alpha_funcion(alpha):
                        return funcion(xk-alpha*grad)
                    
                    alpha = busquedaDorada(alpha_funcion,epsilon=epsilon2,a=0.0,b=1.0)
                    x_k1 = xk - alpha*grad
                    

                    if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2:
                        
                        terminar = True
                    else:
                        k = k+1
                        xk = x_k1
            return xk

* `gradiente_conjugado(funcion,x,epsilon1,epsilon2,epsilon3)` Método de Fletcher-Reeves

        def gradiente_conjugado(funcion,x,epsilon1,epsilon2,epsilon3):
            #step 1
            x0 = x
            #step 2
            grad = np.array(gradiente(funcion,x))
            s0 = -(grad)
            #step 3
            gama = busquedaDorada(funcion,epsilon1,x0,s0)
            k=1
            sk_min1= s0
            xk = gama
            dev_xk=np.array(gradiente(funcion,xk))
            terminar = 0
            
            while terminar != 1:
                #step 4
                sk = -(dev_xk) + np.dot(np.divide(np.sum(dev_xk)**2,np.sum(grad)**2), sk_min1)
                sk_min1 = sk
                #step5
                gama_xk=busquedaDorada(funcion,0.001,xk,sk)
                # print(gama_xk)
                #step 6
                q1 = (gama_xk[1] - gama_xk[0])/gama_xk[0]
                q2 = np.mean(gama_xk)
                xk = gama_xk
                if (q1/q2) < epsilon2:
                    return sk
                else:
                    terminar = 0
                    k = k+1
                if k == 1000:
                    terminar = 1

* `newton_method(funcion,x0,epsilon1,epsilon2,M)` Método de Newton

        def newton_method(funcion,x0,epsilon1,epsilon2,M):
            # step1
            terminar=False
            xk=x0
            k=0
            while not terminar:
                # step 2
                grad = np.array(gradiente(funcion,xk))
                gradT=np.transpose(grad)
                
                # step 3
                if np.linalg.norm(grad) < epsilon1 or k >= M:
                    terminar=True
                else:

                    def alpha_funcion(alpha):
                        return funcion(xk-alpha*grad)
                    
                    alpha = busquedaDorada(alpha_funcion,epsilon=epsilon2,a=0.0,b=1.0)

                    matrix_H = hessian_matrix(f=funcion,x=xk,deltaX=0.001)
                    Matrix_inv=inv(matrix_H) # matriz hessiana inversa

                    quantity = np.dot(gradT,Matrix_inv)
                    quantity2=np.dot(quantity,grad)
                    
                    # x_k1 = xk - alpha*grad
                    x_k1 = xk-alpha*quantity
                    
                    #step5
                    if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2:
                        terminar = True
                    else:
                        k = k+1
                        xk = x_k1
            return xk

##Funciones objetivo
* Ackley_function
* Beale_function
* Booth_function
* Bukin_functionN6
* Crossintray_function
* Easom_function
* Eggholder_function
* Goldstein_price_function
* Himmelblaus_function
* Holder_table_function
* Levi_functionN13
* Matyas_function
* McCormick_function
* Rastrigin
* Rosenbrock_funt
* Schaffer_functionN2
* Schaffer_functionN4
* Shekel
* Sphere_function
* StyblinskiTang_function
* Three_hump_camel_function


## Ejemplo para mandar a llamar las funciones 

    from metod_univariable import metodos_elim_regiones
    from metod_univariable import metodos_basado_derivada

    from metod_multivariable import metodos_directos
    from metod_multivariable import metodos_gradiente

    from funciones_prueba import func_objetivo
    import numpy as np
    x = np.array([1.0,3.0])
    print(func_objetivo.sphere_function(X=x))

    print(metodos_directos.simplex_search_meth(x,func_objetivo.sphere_function))


