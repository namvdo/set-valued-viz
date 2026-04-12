export const RESERVED_PARAM_NAMES = new Set([
  'x',
  'y',
  'sin',
  'cos',
  'tan',
  'abs',
  'sqrt',
  'exp',
  'ln'
]);

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const validateParamName = (name) => {
  if (!name) return 'Name required';
  if (!NAME_RE.test(name)) {
    return 'Use letters, digits, underscore; start with letter/underscore';
  }
  if (RESERVED_PARAM_NAMES.has(name)) {
    return 'Reserved name';
  }
  return null;
};

export const normalizeParams = (params) => {
  const normalized = params.map(p => ({
    name: (p.name || '').trim(),
    value: Number.isFinite(p.value) ? p.value : Number(p.value)
  }));
  const errors = normalized.map(() => null);
  const seen = new Map();

  normalized.forEach((p, idx) => {
    const nameError = validateParamName(p.name);
    if (nameError) {
      errors[idx] = nameError;
      return;
    }

    if (!Number.isFinite(p.value)) {
      errors[idx] = 'Value must be finite';
      return;
    }

    if (seen.has(p.name)) {
      const firstIdx = seen.get(p.name);
      errors[idx] = 'Duplicate name';
      if (!errors[firstIdx]) errors[firstIdx] = 'Duplicate name';
    } else {
      seen.set(p.name, idx);
    }
  });

  const valid = errors.every(err => !err);
  return { normalized, errors, valid };
};

export const formatParamSummary = (params, max = 3) => {
  if (!params.length) return 'no params';
  const shown = params.slice(0, max).map(p => `${p.name}=${p.value.toFixed(3)}`);
  const suffix = params.length > max ? `, +${params.length - max} more` : '';
  return `${shown.join(', ')}${suffix}`;
};
