import os

def procesar_archivo(input_path, output_path):
    try:
        # Verificamos que el archivo de entrada exista
        if not os.path.exists(input_path):
            print(f"Error: El archivo '{input_path}' no existe.")
            return

        print(f"Procesando {input_path}...")
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            count = 0
            for line in f_in:
                # Eliminamos el salto de línea y espacios al final para procesar
                stripped_line = line.rstrip()
                
                # Si la línea está vacía, la saltamos o la dejamos igual (opcional)
                if not stripped_line:
                    continue

                # Lógica de condiciones
                # Nota: Usamos 'elif' para que solo se añada un número por línea
                # si hay conflicto de palabras.
                suffix = ""
                
                if 'learned' in stripped_line:
                    suffix = ",6"
                elif 'blank' in stripped_line:
                    suffix = ",0"
                elif 'coords' in stripped_line:
                    suffix = ",2"
                elif 'hybrid' in stripped_line:
                    suffix = ",8"
                
                # Escribimos la línea original + el sufijo + el salto de línea
                f_out.write(stripped_line + suffix + '\n')
                count += 1

        print(f"¡Listo! Se procesaron {count} líneas.")
        print(f"Archivo guardado en: {output_path}")

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# --- Configuración ---
# Cambia 'entrada.csv' por la ruta de tu archivo
archivo_entrada = 'results/PAPER_TSP50.csv'
archivo_salida = 'results/evaluations/PAPER_TSP50_processed.csv'

if __name__ == "__main__":
    procesar_archivo(archivo_entrada, archivo_salida)