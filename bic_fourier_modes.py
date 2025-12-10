# Imports
import numpy as np
import matplotlib.pyplot as plt

#Observation window length
T = t.max() - t.min()
N = len(t) #Total number of data points

def calculate_bic(ssr, N, k):
    # The term N * np.log(ssr / N) is equivalent to N * np.log(sigma_hat^2)
    return N * np.log(ssr / N) + k * np.log(N)

def calculate_ssr_from_fourier_fit(t, signal, sigma, B):
    """
    Fits the Fourier series model for a given B, and calculates SSR.
    We are using the provided Bayesian framework (mu_post is the mean fit).
    """
    T = t.max() - t.min()
    X = build_design_matrix(t, B, T)
    p = X.shape[1] # Number of parameters for mode B (2B + 1)

    tau2 = 1e4
    invSigma0 = np.eye(p) / tau2

    #Observational precision matrix W
    W = np.diag(1.0 / (sigma**2))

    #Posteriors
    XTW_X = X.T @ W @ X
    XTW_y = X.T @ W @ signal

    # P\osterior precision and covariance
    post_precision = XTW_X + invSigma0
   
    Sigma_post = np.linalg.inv(post_precision) 
    mu_post = Sigma_post @ XTW_y

    #Model prediction at the data points
    model_mean = X @ mu_post

    #Sum of Squared Residuals (SSR)
    residuals = signal - model_mean
    ssr = np.sum(residuals**2)

    return ssr, p

def find_best_truncation_bic_multiple(t, signal, sigma, max_modes=20, bic_tolerance=2.0):
    """
    Finds the best Fourier modes (B) using BIC, and returns all modes 
    whose BIC is within a specified tolerance of the minimum BIC.
    
    bic_tolerance: The maximum BIC difference from the minimum accepted 
                   for a mode to be considered "good." (e.g., 2.0 or 6.0)
    """
    N = len(t)
    results = {}  # Dictionary to store {B: BIC} for all modes

    # --- 1. Calculate BIC for B=0 (DC term only) ---
    mean_signal = np.mean(signal)
    ssr_0 = np.sum((signal - mean_signal)**2)
    k_0 = 1 
    bic_0 = calculate_bic(ssr_0, N, k_0)
    results[0] = bic_0
    
    # --- 2. Calculate BICs for B=1 up to max_modes ---
    modes = list(range(1, max_modes + 1))
    
    print(f"Testing modes B=0 to B={max_modes}...")
    
    for B in modes:
        try:
            # Get SSR and number of parameters (k = 2B + 1)
            ssr, k = calculate_ssr_from_fourier_fit(t, signal, sigma, B) 
            bic = calculate_bic(ssr, N, k)
            results[B] = bic
            print(f"Modes (B) {B}: BIC = {bic:.2f}")
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix encountered at B={B}. Stopping mode calculation.")
            break

    # --- 3. Determine the Absolute Best Mode ---
    if not results:
        print("No successful BIC calculations.")
        return [], 0, {}

    # Find the B with the overall minimum BIC
    best_mode = min(results, key=results.get)
    min_bic = results[best_mode]

    # --- 4. Select Modes within the Tolerance ---
    # This selects all B whose BIC is less than or equal to (min_bic + tolerance)
    good_modes = [
        B for B, bic in results.items() 
        if bic <= (min_bic + bic_tolerance)
    ]
    
    # Sort the good modes and their corresponding BICs
    good_modes.sort()
    
    print("-" * 40)
    print(f"Absolute Best Mode: B={best_mode} (BIC: {min_bic:.2f})")
    print(f"Good Modes (within BIC + {bic_tolerance:.1f}): {good_modes}")
    print("-" * 40)

    # Return the list of good modes and the full results dictionary
    return good_modes, best_mode, results

good_modes, overall_best, full_results = find_best_truncation_bic_multiple(t, signal, sigma, max_modes=50, bic_tolerance=25.0)

sorted_items = sorted(full_results.items())
modes_to_plot = [item[0] for item in sorted_items]
bics = [item[1] for item in sorted_items]        

min_bic_value = full_results[overall_best]

plt.figure(figsize=(10, 6))

# Plot the full BIC curve
plt.plot(modes_to_plot, bics, 'o-', color='tab:blue', label='BIC Value')

plt.axvline(overall_best, color='red', linestyle='--', linewidth=2, label=f'Overall Best Mode (B={overall_best})')

good_modes_set = set(good_modes)
for mode in good_modes:
    if mode != overall_best:
        plt.axvline(
            mode, 
            color='green', 
            linestyle=':', 
            alpha=0.7, 
        )


plt.title('Bayesian Information Criterion (BIC) vs. Fourier Mode Truncation (B)')
plt.xlabel('Number of Fourier Modes (B)')
plt.ylabel(r'BIC Value') 
plt.xticks(np.arange(0, np.max(modes_to_plot) + 1, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()