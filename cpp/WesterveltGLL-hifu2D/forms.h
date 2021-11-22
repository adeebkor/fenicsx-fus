// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCx version 0.3.1.dev0.
//
// This code was generated with the following parameters:
//
//  {'assume_aligned': -1,
//   'epsilon': 1e-14,
//   'output_directory': '.',
//   'padlen': 1,
//   'profile': False,
//   'scalar_type': 'double',
//   'table_atol': 1e-09,
//   'table_rtol': 1e-06,
//   'tabulate_tensor_void': False,
//   'ufl_file': ['forms.ufl'],
//   'verbosity': 30,
//   'visualise': False}


#pragma once

#include <ufc.h>

#ifdef __cplusplus
extern "C" {
#endif

extern ufc_finite_element element_ee7755fb9192d85e9bbabdfa958a19575d037fcb;

extern ufc_finite_element element_c7ad61b539c6dd97225dd4dda3c3916557532058;

<<<<<<< HEAD
extern ufc_finite_element element_a5e99f850cd469e7d420d8a3568af11c4cdb6db9;
=======
extern ufc_finite_element element_36703e9a146d451316c21a918650aa0b23c1fbe6;
>>>>>>> 3c2105645a2178d06a12ac452d12cdcb13ddb906

extern ufc_dofmap dofmap_ee7755fb9192d85e9bbabdfa958a19575d037fcb;

extern ufc_dofmap dofmap_c7ad61b539c6dd97225dd4dda3c3916557532058;

<<<<<<< HEAD
extern ufc_dofmap dofmap_a5e99f850cd469e7d420d8a3568af11c4cdb6db9;

extern ufc_integral integral_e4cf9813e901fd4b2dbf04ca5092ebe5bd542e88;

extern ufc_integral integral_18e8df3b63f23eb818d0522d75113fbc6947e708;

extern ufc_integral integral_fd61fc261139bac354d4b276bbbf65a8ee913765;

extern ufc_integral integral_1d4700c60b3ca0cfecddc6f02b4f949e3a71c475;

extern ufc_integral integral_a7da48916595f2431cffb617783be90616dc92e3;

extern ufc_form form_86377cd6345af7ba4d01b81dad2dbf31eab26530;
=======
extern ufc_dofmap dofmap_36703e9a146d451316c21a918650aa0b23c1fbe6;

extern ufc_integral integral_5879324e980af001706126dd8068de10204532fb;

extern ufc_integral integral_24268f8e2bb0ad053e4ffa1866a21b04badf6be1;

extern ufc_integral integral_a6ff448089b6a76546fb6d0d616325c019e36e82;

extern ufc_integral integral_1d7a1d6934ff9987ae08b3acf1727ac5251ba06e;

extern ufc_integral integral_6305e1d6b7114acb3b886148a2dfca169fa85172;

extern ufc_form form_4397f51aaed49c3cea67854428fee37c0dbb6557;
>>>>>>> 3c2105645a2178d06a12ac452d12cdcb13ddb906

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_forms_a;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_forms_a(const char* function_name);

<<<<<<< HEAD
extern ufc_form form_0a51d3d8198a4b5da38178bcc61e76a100d11ea4;
=======
extern ufc_form form_05af79ec8388ceb9d5af037ecce1053a7fd0e421;
>>>>>>> 3c2105645a2178d06a12ac452d12cdcb13ddb906

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_forms_L;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_forms_L(const char* function_name);

#ifdef __cplusplus
}
#endif
