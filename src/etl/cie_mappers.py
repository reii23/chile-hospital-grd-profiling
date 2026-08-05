"""Mapeadores de códigos clínicos a capítulos/secciones.

Implementa:
- `Mapeador_CIE10`: código CIE-10 (formato letra+dígitos, e.g. "A09.0", "I21.4")
  → capítulo `cap_01..cap_22` o `CAP_DESCONOCIDO`.
- `Mapeador_CIE9`: código CIE-9-MC (numérico decimal, e.g. "13.41")
  → sección `sec_NN` (con sufijo 'A' opcional) o `SEC_DESCONOCIDA`.

Las funciones de normalización (Req 15.4) son **idempotentes** y deterministas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CAP_DESCONOCIDO = "CAP_DESCONOCIDO"
SEC_DESCONOCIDA = "SEC_DESCONOCIDA"

# Regex para extraer el número de capítulo del texto "Cap.NN  TÍTULO ..."
CIE10_CAPITULO_RE = re.compile(r"^Cap\.(\d{2})\b", re.IGNORECASE)

# Regex para extraer el código de sección CIE-9-MC del texto "NN. (...) TÍTULO" o "NNA. (...) ..."
CIE9_SECCION_RE = re.compile(r"^(\d{2}[A-Z]?)\.")


# ---------------------------------------------------------------------------
# Normalizadores (Req 15.4)
# ---------------------------------------------------------------------------
def normalizar_cie10(codigo) -> str | None:
    """Normaliza un código CIE-10 a forma canónica.

    Operaciones (idempotentes y compuestas):
    1. Convertir a string (None/NaN → None).
    2. Strip (eliminar espacios extremos).
    3. Uppercase.
    4. Eliminar punto final si quedó suelto (ej. "A09." → "A09").

    Examples
    --------
    >>> normalizar_cie10("  a09.0  ")
    'A09.0'
    >>> normalizar_cie10("A09.")
    'A09'
    >>> normalizar_cie10(None)
    >>> normalizar_cie10("")
    """
    if codigo is None:
        return None
    # Manejar NaN de pandas sin importar numpy
    if isinstance(codigo, float) and codigo != codigo:  # NaN check
        return None
    s = str(codigo).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        return None
    return s.rstrip(".")


def normalizar_cie9(codigo) -> str | None:
    """Normaliza un código CIE-9-MC a forma canónica.

    A diferencia de CIE-10, los códigos CIE-9-MC son numéricos y conservan el
    punto decimal interno. La normalización:
    1. Convertir a string (None/NaN → None).
    2. Strip.
    3. Eliminar coma (algunos archivos chilenos usan coma decimal).

    No aplica uppercase porque los códigos son solo dígitos.

    Examples
    --------
    >>> normalizar_cie9("  13.41  ")
    '13.41'
    >>> normalizar_cie9("13,41")
    '13.41'
    >>> normalizar_cie9(13.41)
    '13.41'
    >>> normalizar_cie9(None)
    """
    if codigo is None:
        return None
    if isinstance(codigo, float) and codigo != codigo:  # NaN check
        return None
    s = str(codigo).strip().replace(",", ".")
    if not s or s in {"nan", "None", "NaN"}:
        return None
    return s


# ---------------------------------------------------------------------------
# Mapeador_CIE10 (Req 5.1, 5.2)
# ---------------------------------------------------------------------------
@dataclass
class Mapeador_CIE10:
    """Mapea códigos CIE-10 a capítulos `cap_01..cap_22` o `CAP_DESCONOCIDO`.

    El archivo `CIE-10.xlsx` ya trae la columna `Capítulo` con texto del estilo
    `"Cap.01  CIERTAS ENFERMEDADES INFECCIOSAS Y PARASITARIAS (A00-B99)"`.
    Se extrae el número de capítulo con regex y se descartan filas auxiliares
    (p. ej. "TAB M MORFOLOGÍAS DE LAS NEOPLASIAS", que no es capítulo estándar).

    Attributes
    ----------
    df_maestra : pd.DataFrame
        DataFrame con al menos columnas `Código` y `Capítulo` (formato del Excel
        oficial CIE-10).

    Notes
    -----
    El lookup se construye sobre el código exacto del Excel (incluye códigos
    "padre" como `A00` y todas sus subdivisiones `A00.0`, `A00.1`, etc.). Para
    un código nuevo no presente en la maestra, el mapeo retorna `CAP_DESCONOCIDO`.
    """

    df_maestra: pd.DataFrame
    _lookup: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _capitulos_unicos: list[str] = field(default_factory=list, init=False, repr=False)

    # ----- API pública -----
    def construir(self) -> dict[str, str]:
        """Construye el lookup `código_normalizado → cap_NN`.

        Returns
        -------
        dict[str, str]
            Mapeo. Capítulos en formato `cap_01..cap_22`.
        """
        df = self.df_maestra.copy()
        df["_codigo_norm"] = df["Código"].map(normalizar_cie10)
        df["_cap_label"] = df["Capítulo"].map(self._capitulo_a_label)

        # Descartar filas sin código o sin capítulo válido (e.g. TAB M, header rows)
        df = df.dropna(subset=["_codigo_norm"])
        df = df[df["_cap_label"] != CAP_DESCONOCIDO]

        # En caso de duplicados, conservar la primera aparición
        df = df.drop_duplicates(subset=["_codigo_norm"], keep="first")

        self._lookup = dict(zip(df["_codigo_norm"], df["_cap_label"]))
        self._capitulos_unicos = sorted(df["_cap_label"].unique().tolist())
        return self._lookup

    def mapear(self, codigos: pd.Series) -> pd.Series:
        """Mapea una serie de códigos CIE-10 a sus capítulos.

        Parameters
        ----------
        codigos : pd.Series
            Serie con códigos CIE-10 (pueden tener variaciones de formato).

        Returns
        -------
        pd.Series
            Serie con etiquetas `cap_NN` o `CAP_DESCONOCIDO`. Valores nulos en
            la entrada producen `CAP_DESCONOCIDO`.
        """
        if not self._lookup:
            self.construir()
        normalizados = codigos.map(normalizar_cie10)
        # Lookup vectorizado: el primer match cuenta. Para códigos derivados
        # (e.g. "A00.1" no presente en el lookup directo), intentamos el padre
        # eliminando el sufijo decimal.
        result = normalizados.map(self._lookup)
        # Fallback: si el código exacto no está, intentar con el padre (sin decimal)
        sin_match = result.isna() & normalizados.notna()
        if sin_match.any():
            padres = normalizados[sin_match].str.split(".").str[0]
            result.loc[sin_match] = padres.map(self._lookup)
        return result.fillna(CAP_DESCONOCIDO)

    @property
    def capitulos(self) -> list[str]:
        """Lista ordenada de capítulos disponibles (sin contar `CAP_DESCONOCIDO`)."""
        if not self._lookup:
            self.construir()
        return self._capitulos_unicos

    # ----- helpers privados -----
    @staticmethod
    def _capitulo_a_label(texto) -> str:
        """Convierte 'Cap.09  ENFERMEDADES DEL APARATO CIRCULATORIO (I00-I99)'
        en 'cap_09'. Si no coincide el patrón, retorna `CAP_DESCONOCIDO`.
        """
        if texto is None or (isinstance(texto, float) and texto != texto):
            return CAP_DESCONOCIDO
        m = CIE10_CAPITULO_RE.match(str(texto).strip())
        if not m:
            return CAP_DESCONOCIDO
        return f"cap_{int(m.group(1)):02d}"


# ---------------------------------------------------------------------------
# Mapeador_CIE9 (Req 6.1, 6.2)
# ---------------------------------------------------------------------------
@dataclass
class Mapeador_CIE9:
    """Mapea códigos CIE-9-MC a secciones `sec_NN` o `SEC_DESCONOCIDA`.

    Las secciones CIE-9-MC se definen por **rangos enteros** (p. ej. la sección
    "Operaciones sobre el ojo" cubre los códigos 08–16). Por eso el lookup se
    indexa por la **parte entera** del código (antes del punto).

    Ejemplo: `13.41` → entero `13` → sección que cubre `13` → `sec_13` (procedimientos obstétricos).

    Attributes
    ----------
    df_maestra : pd.DataFrame
        DataFrame con columnas `Código` y `Sección` (formato del Excel oficial CIE-9-MC).
    """

    df_maestra: pd.DataFrame
    _lookup: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _secciones_unicas: list[str] = field(default_factory=list, init=False, repr=False)

    def construir(self) -> dict[int, str]:
        """Construye el lookup `entero → sec_NN`.

        Para cada entero presente en la columna `Código` (parte antes del punto),
        guarda la etiqueta de sección correspondiente. Si dos códigos del mismo
        entero pertenecen a secciones distintas (raro, pero posible), conserva
        la primera aparición.

        Returns
        -------
        dict[int, str]
            Mapeo `entero → sec_NN[A]` (con sufijo opcional para "03A").
        """
        df = self.df_maestra.copy()
        df["_codigo_str"] = df["Código"].astype(str).str.strip().str.replace(",", ".")
        df["_entero"] = pd.to_numeric(
            df["_codigo_str"].str.split(".").str[0], errors="coerce"
        )
        df["_sec_label"] = df["Sección"].map(self._seccion_a_label)

        df = df.dropna(subset=["_entero"])
        df = df[df["_sec_label"] != SEC_DESCONOCIDA]
        df["_entero"] = df["_entero"].astype(int)
        df = df.drop_duplicates(subset=["_entero"], keep="first")

        self._lookup = dict(zip(df["_entero"], df["_sec_label"]))
        self._secciones_unicas = sorted(df["_sec_label"].unique().tolist())
        return self._lookup

    def mapear(self, codigos: pd.Series) -> pd.Series:
        """Mapea una serie de códigos CIE-9-MC a sus secciones.

        Parameters
        ----------
        codigos : pd.Series
            Serie con códigos CIE-9-MC (numéricos o strings tipo "13.41").

        Returns
        -------
        pd.Series
            Serie con etiquetas `sec_NN` o `SEC_DESCONOCIDA`. Valores nulos
            producen `SEC_DESCONOCIDA`.
        """
        if not self._lookup:
            self.construir()
        normalizados = codigos.map(normalizar_cie9)
        enteros = normalizados.str.split(".").str[0]
        # `pd.to_numeric` convierte a NaN los strings no numéricos; -1 es sentinel
        idx = pd.to_numeric(enteros, errors="coerce").fillna(-1).astype(int)
        return idx.map(self._lookup).fillna(SEC_DESCONOCIDA)

    @property
    def secciones(self) -> list[str]:
        """Lista ordenada de secciones disponibles (sin contar `SEC_DESCONOCIDA`)."""
        if not self._lookup:
            self.construir()
        return self._secciones_unicas

    # ----- helpers privados -----
    @staticmethod
    def _seccion_a_label(texto) -> str:
        """Convierte '13. (72 75) PROCEDIMIENTOS OBSTETRICOS' en 'sec_13'.
        Convierte '03A. (17 17) ...' en 'sec_03A'.
        """
        if texto is None or (isinstance(texto, float) and texto != texto):
            return SEC_DESCONOCIDA
        m = CIE9_SECCION_RE.match(str(texto).strip())
        if not m:
            return SEC_DESCONOCIDA
        return f"sec_{m.group(1)}"
