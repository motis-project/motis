module.exports = {
  root: true,
  env: {
    node: true
  },
  'extends': [
    'plugin:vue/vue3-strongly-recommended',
    'eslint:recommended',
    '@vue/typescript/recommended'
  ],
  parserOptions: {
    parser: "@typescript-eslint/parser",
    ecmaVersion: 2020,
    ecmaFeatures : {
      jsx : false
    }
  },
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    "vue/attribute-hyphenation":  ["error", "never", {
      "ignore": []
    }],
    "vue/html-closing-bracket-newline": ["warn", {
      "singleline": "never",
      "multiline": "never"
    }],
    "vue/html-self-closing": ["error", {
      "html": {
        "void": "always",
        "normal": "never",
        "component": "never"
      },
      "svg": "never",
      "math": "never"
    }],
    "vue/max-attributes-per-line": ["warn", {
      "singleline": {
        "max": 3
      },
      "multiline": {
        "max": 1
      }
    }],
    "vue/v-on-event-hyphenation": ["warn", "never", {
      "autofix": true,
      "ignore": []
    }],
    "vue/no-v-html": "error",
    "vue/component-tags-order": "error",
    "vue/order-in-components": "warn",
    "vue/this-in-template": "error",
    "vue/block-lang": ["error",
    {
      "script": {
        "lang": "ts"
      }
    }],
    "vue/component-name-in-template-casing": "warn",
    "vue/custom-event-name-casing": ["warn", "camelCase",
    {
      "ignores": []
    }],
    "vue/match-component-file-name": ["error", {
      "extensions": ["vue"],
      "shouldMatchCase": true
    }],
    "vue/no-v-text": "error",
    "vue/no-unregistered-components": "error",
    "vue/padding-line-between-blocks": "warn",
    "vue/require-direct-export": "error",
    "vue/require-name-property": "error",
    "vue/script-indent": "warn",
    "vue/v-for-delimiter-style": "error",
    "vue/require-default-prop": "off",
    "eqeqeq": "error",
    "no-duplicate-imports": "warn",
    "no-self-compare": "error",
    "brace-style": ["warn", "stroustrup", { "allowSingleLine": true }],
    "camelcase": "warn",
    "curly": "warn",
    "no-trailing-spaces": "warn",
    "default-param-last": "error",
    "dot-notation": "warn",
    "no-empty": "off",
    "no-lonely-if": "warn",
    "no-multi-assign": "error",
    "no-useless-escape": "off",
    "no-var": "warn",
    "require-await": "error",
    "block-spacing": "warn",
    "comma-spacing": "warn",
    "eol-last": "warn",
    "func-call-spacing": "warn",
  }
}
